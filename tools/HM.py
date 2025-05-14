import os
import csv
import cv2
import math
import torch
import numpy as np
import pandas as pd

from torch.nn.functional import cosine_similarity
from torchreid.utils import FeatureExtractor
from collections import defaultdict, Counter
from tqdm import tqdm


# -----------------------------
#  1) HÀM PHỤ: LOAD REID MODEL
# -----------------------------
def load_reid_model(
        model_name='osnet_ain_x1_0',
        model_path='osnet_x1_0.pth',
        use_cuda=True
):
    device = 'cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu'
    model = FeatureExtractor(
        model_name=model_name,
        model_path=model_path,
        device=device
    )
    return model


# -------------------------------------------------
#  2) TÍNH KHOẢNG CÁCH: COSINE + KHOẢNG CÁCH EUCLID
# -------------------------------------------------
def compute_fusion_distance(feat1, feat2, x3d1, y3d1, x3d2, y3d2, lamda=0.7):
    """
    - feat1, feat2: torch.Tensor (D,) đặc trưng ReID
    - x3d1, y3d1, x3d2, y3d2: tọa độ 3D (Z=0)
    - lamda: trọng số
    Return: fusion_distance = lamda * cosine_dist + (1 - lamda) * euclid_dist
    """
    if feat1 is None or feat2 is None:
        # Nếu thiếu feature, trả về khoảng cách lớn để tránh gộp sai
        cos_dist = 1.0
    else:
        # Bước 1: Normalize vector
        feat1_norm = feat1 / (feat1.norm(p=2) + 1e-8)
        feat2_norm = feat2 / (feat2.norm(p=2) + 1e-8)

        # Bước 2: Tính cosine similarity
        cos_sim = cosine_similarity(feat1_norm.unsqueeze(0), feat2_norm.unsqueeze(0)).item()

        # Bước 3: Clamp giá trị trong [-1, 1]
        cos_sim = min(max(cos_sim, -1.0), 1.0)

        # Bước 4: Tính khoảng cách cosine
        cos_dist = 1.0 - cos_sim

    dx = x3d1 - x3d2
    dy = y3d1 - y3d2
    euc_dist = math.sqrt(dx * dx + dy * dy)

    fusion_dist = lamda * cos_dist + (1 - lamda) * euc_dist
    return fusion_dist


# -----------------------------
#  3) PHÂN CỤM BẰNG H.CLUSTER
# -----------------------------
def hierarchical_clustering(objects, fusion_dist_threshold=0.7, max_cluster_size=4, lamda=0.7):
    """
    objects: list chứa dict,
        mỗi dict = {
          'feature': feat_vector (torch.Tensor hoặc None),
          'x3d': float,
          'y3d': float,
          'local_id': assigned_id (cũ)
        }
    Trả về cluster_labels (list nhãn cụm, cùng thứ tự với 'objects').
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    N = len(objects)
    if N == 0:
        return []
    if N == 1:
        return [1]  # chỉ 1 đối tượng, cụm duy nhất

    # Tạo ma trận NxN
    dist_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            dist = compute_fusion_distance(
                objects[i]['feature'],
                objects[j]['feature'],
                objects[i]['x3d'], objects[i]['y3d'],
                objects[j]['x3d'], objects[j]['y3d'],
                lamda=lamda
            )
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist

    # Chuyển ma trận NxN sang dạng condensed
    dist_condensed = squareform(dist_mat, checks=False)

    # Phân cụm hierarchical
    Z = linkage(dist_condensed, method='average')
    labels = fcluster(Z, t=fusion_dist_threshold, criterion='distance')

    # Hàm phụ tách cụm nếu kích thước > max_cluster_size
    def check_and_split(labels, Z, threshold):
        group_map = defaultdict(list)
        for idx, clab in enumerate(labels):
            group_map[clab].append(idx)

        need_split = []
        for clab, idx_list in group_map.items():
            if len(idx_list) > max_cluster_size:
                need_split.append(clab)

        if not need_split:
            return labels

        new_threshold = threshold * 0.9
        if new_threshold < 1e-3:
            return labels  # tránh lặp vô hạn

        new_labels = fcluster(Z, t=new_threshold, criterion='distance')
        return check_and_split(new_labels, Z, new_threshold)

    labels = check_and_split(labels, Z, fusion_dist_threshold)
    return labels


# ----------------------------------------------
#  4) HÀM XỬ LÝ TỪNG FRAME: TÍNH FEATURE REID, ...
# ----------------------------------------------
def process_frame_objects(frame_obj_list, reid_model, frame_img):
    """
    frame_obj_list: list các dict { 'camera_id', 'assigned_id', 'x1','y1','x2','y2', 'x3d','y3d' }
    frame_img: ảnh (ndarray) của camera tương ứng
    reid_model: model ReID

    Trích xuất crop tương tự CC_singlecam.py => tính feature
    Trả về list dict { 'feature', 'x3d', 'y3d', 'local_id' }
    """
    results = []
    h_img, w_img = frame_img.shape[:2]

    for obj in frame_obj_list:
        # Cắt bounding box
        x1 = int(obj['x1'])
        y1 = int(obj['y1'])
        x2 = int(obj['x2'])
        y2 = int(obj['y2'])

        # Chặn lại để tránh out-of-bound
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img - 1, x2)
        y2 = min(h_img - 1, y2)

        if x2 <= x1 or y2 <= y1:
            feat = None
        else:
            crop = frame_img[y1:y2, x1:x2]
            if crop.size == 0:
                feat = None
            else:
                feat = reid_model([crop])[0].cpu()

        results.append({
            'feature': feat,
            'x3d': float(obj['x3d']),
            'y3d': float(obj['y3d']),
            'local_id': obj['assigned_id']
        })

    return results


# -----------------------------------
#  5) TỔNG HỢP XỬ LÝ TOÀN BỘ 4 CAMERA
# -----------------------------------
def multi_camera_global_tracking(
        input_csv='global_final_result.csv',
        output_csv='global_final_result_improve.csv',
        videos={
            0: 'cam131.avi',
            1: 'cam132.avi',
            2: 'cam133.avi',
            3: 'cam134.avi',
        },
        reid_model_path='osnet_x1_0.pth',
        lamda=0.7,
        fusion_dist_threshold=0.5,
        min_obj_dis_threshold=0.15,
        tracklet_inactive_threshold=5  # số frame tối đa cho phép tracklet không được cập nhật
):
    """
    - Các bước: đọc CSV, mở video, trích xuất đối tượng, tính feature, phân cụm cho từng frame.
    - Sửa đổi: Tích hợp cơ chế tracklet để đảm bảo sự liên tục của global_id qua nhiều frame.
      Phần cập nhật tracklet được thay đổi theo nguyên tắc majority voting với chuỗi vote gồm (2*tracklet_inactive_threshold + 1) frame.
    """
    # Đọc CSV
    df = pd.read_csv(input_csv)
    df['global_id'] = -1

    # Load model ReID một lần
    reid_model = load_reid_model(
        model_name='osnet_ain_x1_0',
        model_path=reid_model_path,
        use_cuda=True
    )

    # Mở video cho từng camera
    caps = {}
    total_frames_cam = {}
    for cid, vid_path in videos.items():
        if not os.path.isfile(vid_path):
            print(f"WARNING: File video {vid_path} không tồn tại.")
            caps[cid] = None
            total_frames_cam[cid] = 0
            continue
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"WARNING: Không mở được video {vid_path}")
            caps[cid] = None
            total_frames_cam[cid] = 0
            continue
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        caps[cid] = cap
        total_frames_cam[cid] = total_f

    max_frame = df['frame_id'].max()
    clusters_info = {}

    # Khởi tạo active tracklets và biến đếm tracklet
    active_tracklets = {}  # key: tracklet_id, value: dict { 'global_id', 'last_center', 'last_frame', 'votes': { frame_id: provisional_global_id } }
    tracklet_counter = 0

    print("=== Start multi-camera global tracking (với tracklet) ===")
    for frame_id in tqdm(range(1, max_frame + 1)):
        # Lấy các đối tượng của frame hiện tại
        df_frame = df[df['frame_id'] == frame_id]
        if len(df_frame) == 0:
            continue

        all_objects = []
        idx_map = []

        # Đọc frame từ các camera
        for cid, cap in caps.items():
            if cap is None:
                continue
            if frame_id > total_frames_cam[cid]:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
            ret, frame_img = cap.read()
            if not ret:
                continue

            df_cam_frame = df_frame[df_frame['camera_id'] == cid]
            if len(df_cam_frame) == 0:
                continue

            frame_obj_list = df_cam_frame.to_dict('records')
            frame_results = process_frame_objects(frame_obj_list, reid_model, frame_img)
            for i, rr in enumerate(frame_results):
                row_idx = df_cam_frame.index[i]
                all_objects.append(rr)
                idx_map.append(row_idx)

        if len(all_objects) == 0:
            continue

        # Phân cụm cho các đối tượng của frame hiện tại
        cluster_labels = hierarchical_clustering(
            all_objects,
            fusion_dist_threshold=fusion_dist_threshold,
            max_cluster_size=4,
            lamda=lamda
        )

        # Xây dựng current_clusters với thông tin ban đầu: majority của local_id và tính center
        current_clusters = []
        cluster_map = defaultdict(list)
        for obj_idx, c_label in enumerate(cluster_labels):
            cluster_map[c_label].append(obj_idx)
        for c_label, obj_indices in cluster_map.items():
            local_ids = [all_objects[i]['local_id'] for i in obj_indices]
            count_local = Counter(local_ids)
            init_global_id, _ = count_local.most_common(1)[0]
            xs = [all_objects[i]['x3d'] for i in obj_indices]
            ys = [all_objects[i]['y3d'] for i in obj_indices]
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            current_clusters.append({
                'global_id': init_global_id,  # khởi tạo từ majority local id
                'center': (center_x, center_y),
                'object_indices': obj_indices,
                'cluster_label': c_label
            })

        # -----------------------------------------------
        # Phần cập nhật tracklet: cập nhật global_id theo majority voting
        # với chuỗi vote gồm (2*tracklet_inactive_threshold + 1) frame
        # (ví dụ: với tracklet_inactive_threshold=5 => 11 frame: 5 frame trước, frame hiện tại, 5 frame sau)
        # -----------------------------------------------
        for cluster in current_clusters:
            best_tracklet_id = None
            best_distance = float('inf')
            # Tìm tracklet phù hợp từ active_tracklets
            for tid, tdata in active_tracklets.items():
                if frame_id - tdata['last_frame'] > tracklet_inactive_threshold:
                    continue
                dist = math.sqrt((cluster['center'][0] - tdata['last_center'][0])**2 + 
                                 (cluster['center'][1] - tdata['last_center'][1])**2)
                if dist < best_distance and dist < min_obj_dis_threshold:
                    best_distance = dist
                    best_tracklet_id = tid
            if best_tracklet_id is not None:
                # Cập nhật vote history cho tracklet đã tìm được
                active_tracklets[best_tracklet_id]['votes'][frame_id] = cluster['global_id']
                # Cập nhật thông tin center và frame của tracklet
                active_tracklets[best_tracklet_id]['last_center'] = cluster['center']
                active_tracklets[best_tracklet_id]['last_frame'] = frame_id
                # Lấy các vote trong cửa sổ [frame_id - T, frame_id + T]
                T = tracklet_inactive_threshold
                window_votes = [vote for f, vote in active_tracklets[best_tracklet_id]['votes'].items() if (frame_id - T) <= f <= (frame_id + T)]
                if window_votes:
                    maj_vote = Counter(window_votes).most_common(1)[0][0]
                else:
                    maj_vote = active_tracklets[best_tracklet_id]['global_id']
                # Cập nhật global_id theo majority vote
                cluster['global_id'] = maj_vote
                active_tracklets[best_tracklet_id]['global_id'] = maj_vote
            else:
                # Nếu không tìm được tracklet phù hợp, tạo tracklet mới với vote history khởi tạo
                tracklet_counter += 1
                new_tracklet_id = tracklet_counter
                active_tracklets[new_tracklet_id] = {
                    'global_id': cluster['global_id'],
                    'last_center': cluster['center'],
                    'last_frame': frame_id,
                    'votes': {frame_id: cluster['global_id']}
                }
        # -----------------------------------------------

        # Ghi global_id cho từng đối tượng theo cụm vào DataFrame
        for cluster in current_clusters:
            for idx_in_cluster in cluster['object_indices']:
                row_idx = idx_map[idx_in_cluster]
                df.at[row_idx, 'global_id'] = cluster['global_id']

        # Lưu thông tin clusters cho frame hiện tại
        clusters_info[frame_id] = []
        for cluster in current_clusters:
            clusters_info[frame_id].append({
                'global_id': cluster['global_id'],
                'center': cluster['center']
            })

        # Loại bỏ các tracklet không được cập nhật trong nhiều frame
        inactive_ids = []
        for tid, tdata in active_tracklets.items():
            if frame_id - tdata['last_frame'] > tracklet_inactive_threshold:
                inactive_ids.append(tid)
        for tid in inactive_ids:
            del active_tracklets[tid]

    # Ghi kết quả vào file CSV
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Done. Kết quả đã ghi vào {output_csv}")

    return clusters_info


# ---------------
# MAIN (DEMO)
# ---------------
if __name__ == '__main__':
    clusters_info = multi_camera_global_tracking(
        input_csv='singlecam_results_new/leaf2/global_final_result.csv',
        output_csv='singlecam_results_new/leaf2/global_final_result_acivs.csv',
        videos={
            0: '/content/drive/MyDrive/NCKH_MultiCamera/ICGLab6/leaf2/cam131.avi',
            1: '/content/drive/MyDrive/NCKH_MultiCamera/ICGLab6/leaf2/cam132.avi',
            2: '/content/drive/MyDrive/NCKH_MultiCamera/ICGLab6/leaf2/cam133.avi',
            3: '/content/drive/MyDrive/NCKH_MultiCamera/ICGLab6/leaf2/cam134.avi'
        },
        reid_model_path='/content/drive/MyDrive/NCKH_MultiCamera/models_/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth',
        lamda=0.7,
        fusion_dist_threshold=0.5,
        min_obj_dis_threshold=0.15,
        tracklet_inactive_threshold=7
    )
