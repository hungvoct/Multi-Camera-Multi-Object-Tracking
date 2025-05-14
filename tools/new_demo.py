import os
import os.path as osp
import cv2
import torch
from loguru import logger
import time
import numpy as np
from collections import defaultdict

# YOLOX + ByteTrack imports
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, xyxy2xywh
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from yolox.data.data_augment import preproc

# Import DBSCAN (cần cài đặt scikit-learn: pip install scikit-learn)
from sklearn.cluster import DBSCAN

####################################
# CẤU HÌNH
####################################
VIDEO_PATH = "/home/your_path/video.avi"    # Thay đổi đường dẫn video của bạn
EXP_FILE = "exps/example/mot/yolox_x_mix_det.py"
MODEL_CKPT = "/home/your_path/bytetrack_x_mot17.pth.tar"

OUTPUT_CSV = "tracking_output_stage1.csv"
OUTPUT_VIDEO = "result_stage1.mp4"
OUTPUT_STRONG_ANCHOR_CSV = "strong_anchor_output.csv"

CONF_THRESH = 0.5   # ngưỡng confidence cho YOLOX
NMS_THRESH = 0.4    # ngưỡng NMS
TRACK_THRESH = 0.5  # ngưỡng cho ByteTrack
TRACK_BUFFER = 30
MATCH_THRESH = 0.8
ASPECT_RATIO_THRESH = 1.6
MIN_BOX_AREA = 10

TEST_IMG_SIZE = (1088, 608)  # kích thước ảnh test (theo exp gốc)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Các tham số cho phân cụm
CLUSTER_EPS = 50           # khoảng cách (pixel) tối đa để 2 điểm cùng cụm
CLUSTER_MIN_SAMPLES = 2    # số điểm tối thiểu trong 1 cụm
CLUSTER_MATCH_DIST = 50    # khoảng cách để so sánh trung tâm của cụm giữa frame t-1 và t
STRONG_ANCHOR_TRACK_LEN = 30   # số frame tối thiểu để đánh giá một track là strong anchor

####################################
# HÀM HỖ TRỢ
####################################
def write_csv_results(csv_path, all_results):
    """
    Ghi kết quả ra file CSV với header:
    frame_id, track_id, x1, y1, w, h, score, state
    """
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "track_id", "x1", "y1", "w", "h", "score", "state"])
        for row in all_results:
            writer.writerow(row)
    logger.info(f"[INFO] Results saved to {csv_path}")

def minimal_inference(model, img, test_size, device):
    """
    Tiền xử lý ảnh, chạy YOLOX inference và postprocess
    Trả về outputs có shape [N, 7] và kích thước gốc (h0, w0)
    """
    h0, w0 = img.shape[:2]
    rgb_means = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    inp, _ = preproc(img, test_size, rgb_means, std)
    inp = torch.from_numpy(inp).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inp)
        outputs = postprocess(outputs,
                              num_classes=80,
                              conf_thre=CONF_THRESH,
                              nms_thre=NMS_THRESH)
    return outputs, (h0, w0)

####################################
# PIPELINE TRACKING + CLUSTERING + STRONG ANCHOR
####################################
def run_tracker_with_clustering():
    logger.info("=== Khởi tạo YOLOX & ByteTrack với pipeline phân cụm và strong anchor ===")
    
    # 1) Tạo exp YOLOX
    exp = get_exp(EXP_FILE, None)
    exp.test_size = TEST_IMG_SIZE
    exp.test_conf = CONF_THRESH
    exp.nmsthre = NMS_THRESH

    # 2) Tạo model, load weight và fuse model
    device = torch.device(DEVICE)
    model = exp.get_model().to(device)
    model.eval()
    logger.info("=> Loading checkpoint")
    ckpt = torch.load(MODEL_CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    logger.info("Done loading checkpoint")
    logger.info("=> Fusing model ...")
    model = fuse_model(model)
    logger.info("=> Model info: {}".format(get_model_info(model, exp.test_size)))

    # 3) Cấu hình ByteTrack
    class SimpleArgs:
        track_thresh = TRACK_THRESH
        track_buffer = TRACK_BUFFER
        match_thresh = MATCH_THRESH
        aspect_ratio_thresh = ASPECT_RATIO_THRESH
        min_box_area = MIN_BOX_AREA
        mot20 = False
        fps = 30  # fps video
    tracker_args = SimpleArgs()
    tracker = BYTETracker(tracker_args, frame_rate=tracker_args.fps)

    # 4) Mở video
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video info: {width}x{height}, fps={fps}, total frames={total_frames}")
    
    # 5) Cấu hình ghi video
    vid_writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 25,
        (width, height)
    )
    
    frame_id = 0
    all_results = []           # lưu kết quả tracking (tất cả)
    strong_anchor_results = [] # lưu kết quả strong anchor
    timer = Timer()
    
    # Biến lưu cụm của frame trước (t-1)
    prev_clusters = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # 6) Inference YOLOX
        outputs, (h0, w0) = minimal_inference(model, frame, exp.test_size, device)
        
        if outputs[0] is not None:
            # 7) Cập nhật tracker (ByteTrack)
            dets = outputs[0]  # [N,7]: [x1, y1, x2, y2, score, class_id, ...]
            dets_for_tracker = dets[:, :5]
            online_targets = tracker.update(dets_for_tracker, [h0, w0], exp.test_size)

            # Lưu kết quả tracking
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                score = t.score
                state = getattr(t, "state", "Tracked")
                all_results.append([
                    frame_id, tid,
                    round(tlwh[0], 2), round(tlwh[1], 2),
                    round(tlwh[2], 2), round(tlwh[3], 2),
                    round(score, 2),
                    state
                ])

            # 8) Phân cụm theo tâm bounding box
            centers = []      # danh sách tâm (cx, cy)
            track_ids = []    # danh sách track_id tương ứng
            tracklet_lens = []  # danh sách độ dài track (nếu có)
            track_data = {}   # map track_id -> đối tượng track
            for t in online_targets:
                x, y, w, h = t.tlwh
                cx = x + w / 2
                cy = y + h / 2
                centers.append([cx, cy])
                track_ids.append(t.track_id)
                # Nếu đối tượng STrack có thuộc tính tracklet_len (được cập nhật qua mỗi frame) thì dùng nó,
                # nếu không thì gán mặc định là 1.
                tracklet_length = getattr(t, "tracklet_len", 1)
                tracklet_lens.append(tracklet_length)
                track_data[t.track_id] = t

            centers = np.array(centers) if len(centers) > 0 else np.empty((0, 2))
            if centers.shape[0] > 0:
                db = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES).fit(centers)
                labels = db.labels_
                current_clusters = {}
                for label in set(labels):
                    indices = np.where(labels == label)[0]
                    cluster_center = centers[indices].mean(axis=0)
                    current_clusters[label] = {
                        "center": cluster_center,
                        "track_ids": [track_ids[i] for i in indices],
                        "max_tracklet_len": max([tracklet_lens[i] for i in indices])
                    }
            else:
                current_clusters = {}

            # 9) So sánh cụm của frame hiện tại với cụm của frame trước (t-1)
            new_clusters = []
            if prev_clusters is not None:
                for label, cluster in current_clusters.items():
                    found = False
                    for prev_label, prev_cluster in prev_clusters.items():
                        dist = np.linalg.norm(cluster["center"] - prev_cluster["center"])
                        if dist < CLUSTER_MATCH_DIST:
                            found = True
                            break
                    if not found:
                        new_clusters.append(cluster)
            else:
                # Nếu không có cụm của frame trước thì coi tất cả đều là mới
                new_clusters = list(current_clusters.values())

            # 10) Tạo strong anchor: nếu trong cụm mới có track có tracklet_len >= STRONG_ANCHOR_TRACK_LEN
            for cluster in new_clusters:
                if cluster["max_tracklet_len"] >= STRONG_ANCHOR_TRACK_LEN:
                    # Đánh dấu các track trong cụm có đủ tiêu chí là strong anchor
                    for tid in cluster["track_ids"]:
                        t_obj = track_data.get(tid, None)
                        if t_obj is not None and getattr(t_obj, "tracklet_len", 0) >= STRONG_ANCHOR_TRACK_LEN:
                            setattr(t_obj, "is_strong_anchor", True)  # thêm thuộc tính đánh dấu strong anchor
                            x, y, w, h = t_obj.tlwh
                            strong_anchor_results.append([
                                frame_id, tid,
                                round(x, 2), round(y, 2),
                                round(w, 2), round(h, 2),
                                round(t_obj.score, 2),
                                "StrongAnchor"
                            ])
            
            # Cập nhật biến lưu cụm cho frame hiện tại (sẽ dùng cho frame kế tiếp)
            prev_clusters = current_clusters

            # 11) Vẽ bounding box và thông tin tracking (có thể thêm cả thông tin cụm, strong anchor,…)
            plot_im = plot_tracking(
                frame, 
                [t.tlwh for t in online_targets],
                [t.track_id for t in online_targets],
                frame_id=frame_id, fps=1.0 / (timer.average_time+1e-6)
            )
            # Vẽ thêm thông báo “Strong” cho các đối tượng strong anchor
            for t in online_targets:
                if getattr(t, "is_strong_anchor", False):
                    x, y, w, h = t.tlwh
                    cv2.rectangle(plot_im, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
                    cv2.putText(plot_im, "Strong", (int(x), int(y-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            plot_im = frame
        
        vid_writer.write(plot_im)
        if frame_id % 10 == 0:
            logger.info(f"Processed frame {frame_id}/{total_frames} ...")
    
    cap.release()
    vid_writer.release()
    
    # 12) Ghi kết quả ra file CSV
    write_csv_results(OUTPUT_CSV, all_results)
    write_csv_results(OUTPUT_STRONG_ANCHOR_CSV, strong_anchor_results)
    logger.info(f"[INFO] All done. Video saved = {OUTPUT_VIDEO}")
    logger.info(f"[INFO] Tracking results saved to {OUTPUT_CSV}")
    logger.info(f"[INFO] Strong anchor results saved to {OUTPUT_STRONG_ANCHOR_CSV}")

if __name__ == "__main__":
    run_tracker_with_clustering()
