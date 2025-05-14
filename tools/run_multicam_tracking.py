import os
import cv2
import time
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter

# YOLOX inference & tracking
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.data.data_augment import preproc
from yolox.tracking_utils.timer import Timer
from yolox.tracker.byte_tracker_multicam import BYTETracker, load_camera_parameters, pixel_to_world_ground
from demo_track_multicam import Predictor, write_results

# HM (ReID + clustering)
import HM

# ===== THAM SỐ CHÍNH =====
calib_file = "/content/drive/MyDrive/NCKH_MultiCamera/ICGLab6/calibration/calibration.xml"
video_info = [
    ("/content/drive/MyDrive/NCKH_MultiCamera/ICGLab6/leaf2/cam131.avi", 0),
    ("/content/drive/MyDrive/NCKH_MultiCamera/ICGLab6/leaf2/cam132.avi", 1),
    ("/content/drive/MyDrive/NCKH_MultiCamera/ICGLab6/leaf2/cam133.avi", 2),
    ("/content/drive/MyDrive/NCKH_MultiCamera/ICGLab6/leaf2/cam134.avi", 3),
]
output_dir = "results_multicam_online/leaf2"
os.makedirs(output_dir, exist_ok=True)

fusion_dist_threshold = 0.7
min_obj_dis_threshold = 0.15
tracklet_inactive_threshold = 5

# 1) Load camera params
cam_params = {}
for _, cam_idx in video_info:
    cam_params[cam_idx] = load_camera_parameters(calib_file, cam_idx)

# 2) Load ReID model
reid_model = HM.load_reid_model(model_path="/content/drive/MyDrive/NCKH_MultiCamera/models_/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth")

# 3) Load YOLOX exp & model
exp = get_exp("exps/example/mot/yolox_x_mix_det.py", None)
args = type("", (), {})()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dùng thông số mặc định trong demo
args.track_thresh = 0.5; args.match_thresh = 0.8; args.fps = 30

model = exp.get_model().to(args.device)
ckpt = torch.load("pretrained/bytetrack_x_mot17.pth.tar", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

# 4) Khởi tạo Predictor & BYTETracker cho từng camera
predictors = {}
trackers = {}
for video_path, cam_idx in video_info:
    predictors[cam_idx] = Predictor(model, exp, None, None, args.device, fp16=False)
    trackers[cam_idx] = BYTETracker(args, frame_rate=args.fps)

# 5) Mở video
caps = {cam_idx: cv2.VideoCapture(vp) for vp, cam_idx in video_info}

# 6) Kết quả lưu tạm
per_cam_results = {cam_idx: [] for _, cam_idx in video_info}
global_clusters = []
active_tracklets = {}
tracklet_counter = 0
timer = Timer()
frame_id = 0

while True:
    # 6.1 Đọc đồng bộ 4 frame
    frames = {}
    ret_any = False
    for cam_idx, cap in caps.items():
        ret, img = cap.read()
        if ret:
            frames[cam_idx] = img
            ret_any = True
    if not ret_any:
        break
    frame_id += 1

    # --- bắt đầu đo thời gian xử lý frame ---
    t0 = time.time()

    # 6.2 Chạy tracker riêng cho từng camera
    all_objs = []
    for cam_idx, img in frames.items():
        outputs, img_info = predictors[cam_idx].inference(img, timer)
        if outputs[0] is None:
            continue
        online_targets = trackers[cam_idx].update(
            outputs[0],
            [img_info["height"], img_info["width"]],
            exp.test_size
        )
        tlwhs, ids, scores = [], [], []
        for t in online_targets:
            # toạ độ 3D
            x1, y1, x2, y2 = *t.tlbr[:2], *t.tlbr[2:]
            u = (x1 + x2) / 2; v = y2
            X, Y, _ = pixel_to_world_ground(u, v, *cam_params[cam_idx])
            all_objs.append({
                "camera_id": cam_idx,
                "assigned_id": t.track_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "x3d": X, "y3d": Y
            })
            if t.score > args.track_thresh:
                tlwhs.append(t.tlwh)
                ids.append(t.track_id)
                scores.append(t.score)
        per_cam_results[cam_idx].append((frame_id, tlwhs, ids, scores))

    if not all_objs:
        continue

    # 6.3 ReID + clustering
    frame_reid = HM.process_frame_objects(all_objs, reid_model, next(iter(frames.values())))
    labels = HM.hierarchical_clustering(frame_reid, fusion_dist_threshold)

    # 6.4 Xây dựng cụm và gán global ID online
    clusters = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[lbl].append(i)

    current = []
    for lbl, inds in clusters.items():
        cx = frame_reid[inds[0]]["x3d"]
        cy = frame_reid[inds[0]]["y3d"]
        current.append({"label": lbl, "global_id": inds[0], "center": (cx, cy)})

    # majority-vote tracklet
    for c in current:
        best_tid, best_dist = None, float("inf")
        for tid, td in active_tracklets.items():
            if frame_id - td["last_frame"] > tracklet_inactive_threshold:
                continue
            d = np.linalg.norm(np.array(c["center"]) - np.array(td["last_center"]))
            if d < best_dist and d < min_obj_dis_threshold:
                best_dist, best_tid = d, tid
        if best_tid:
            td = active_tracklets[best_tid]
            td["votes"][frame_id] = c["global_id"]
            td["last_center"] = c["center"]; td["last_frame"] = frame_id
            window = [v for f,v in td["votes"].items() if abs(f-frame_id)<=tracklet_inactive_threshold]
            maj = Counter(window).most_common(1)[0][0]
            c["global_id"] = maj; td["global_id"] = maj
        else:
            tracklet_counter += 1
            active_tracklets[tracklet_counter] = {
                "global_id": c["global_id"],
                "last_center": c["center"],
                "last_frame": frame_id,
                "votes": {frame_id: c["global_id"]}
            }

    # 6.5 Lưu kết quả global
    for c in current:
        gid = c["global_id"]; cx, cy = c["center"]
        global_clusters.append((frame_id, gid, cx, cy))
    
     # --- kết thúc đo thời gian xử lý frame ---
    t1 = time.time()
    instant_fps = 1.0 / (t1 - t0)
    print(f"[Frame {frame_id:4d}] Inst FPS: {instant_fps:.2f}")

# 7) Xuất file per-camera
for cam_idx, res in per_cam_results.items():
    fn = os.path.join(output_dir, f"cam{cam_idx}_results.txt")
    write_results(fn, res)

# 8) Xuất file global
df = pd.DataFrame(global_clusters, columns=["frame_id","global_id","center_x","center_y"])
df.to_csv(os.path.join(output_dir, "global_results.csv"), index=False)

print("=== Hoàn thành! Kết quả nằm trong: ", output_dir)
