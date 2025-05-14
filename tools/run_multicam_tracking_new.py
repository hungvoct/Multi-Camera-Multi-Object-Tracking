import os
import cv2
import time
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter
from flask import Flask, Response, render_template_string

# YOLOX imports
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

fusion_dist_threshold = 0.7
min_obj_dis_threshold = 0.15
tracklet_inactive_threshold = 5

# 1) Load camera params
cam_params = {cam_idx: load_camera_parameters(calib_file, cam_idx) for _, cam_idx in video_info}

# 2) Load ReID model
reid_model = HM.load_reid_model(model_path="/content/drive/MyDrive/NCKH_MultiCamera/models_/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth")

# 3) Load YOLOX exp & model
exp = get_exp("exps/example/mot/yolox_x_mix_det.py", None)
args = type("", (), {})()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.track_thresh = 0.5
args.match_thresh = 0.8
args.fps = 30
model = exp.get_model().to(args.device)
ckpt = torch.load("pretrained/bytetrack_x_mot17.pth.tar", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

# 4) Initialize Predictor & BYTETracker for each camera
predictors = {}
trackers = {}
for video_path, cam_idx in video_info:
    predictors[cam_idx] = Predictor(model, exp, None, None, args.device, fp16=False)
    trackers[cam_idx] = BYTETracker(args, frame_rate=args.fps)

# 5) Open video captures
caps = {cam_idx: cv2.VideoCapture(vp) for vp, cam_idx in video_info}

def mosaic_frames(frame_dict):
    """
    Arrange 4 camera frames into a 2x2 mosaic.
    """
    cams = sorted(frame_dict.keys())
    imgs = [frame_dict[i] for i in cams]
    # Ensure all frames have same shape
    h, w, _ = imgs[0].shape
    imgs = [cv2.resize(img, (w, h)) for img in imgs]
    top = np.hstack((imgs[0], imgs[1]))
    bottom = np.hstack((imgs[2], imgs[3]))
    return np.vstack((top, bottom))

app = Flask(__name__)

# Ví dụ, width="800" height="450" hoặc width="50%" height="auto"
HTML_PAGE = """
<html>
<head><title>Multi-camera Tracking</title></head>
<body>
<h1>Multi-camera Multi-object Tracking Results</h1>
<!-- Chỉnh kích thước tại đây -->
<img src="{{ url_for('video_feed') }}" width="800" height="450" />
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate():
    frame_id = 0
    active_tracklets = {}
    tracklet_counter = 0
    timer = Timer()
    while True:
        # Read synchronized frames from all cameras
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

        # 6.2 Per-camera tracking
        all_objs = []
        for cam_idx, img in frames.items():
            outputs, img_info = predictors[cam_idx].inference(img, timer)
            if outputs[0] is None:
                continue
            online_targets = trackers[cam_idx].update(
                outputs[0], [img_info['height'], img_info['width']], exp.test_size
            )
            for t in online_targets:
                x1, y1, x2, y2 = *t.tlbr[:2], *t.tlbr[2:]
                u = (x1 + x2) / 2; v = y2
                X, Y, _ = pixel_to_world_ground(u, v, *cam_params[cam_idx])
                all_objs.append({
                    'camera_id': cam_idx,
                    'assigned_id': t.track_id,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'x3d': X, 'y3d': Y
                })

        if not all_objs:
            continue

        # 6.3 ReID + clustering
        # We use a dummy frame just for feature extraction; HM only uses frame for cropping
        dummy_frame = next(iter(frames.values()))
        frame_reid = HM.process_frame_objects(all_objs, reid_model, dummy_frame)
        labels = HM.hierarchical_clustering(frame_reid, fusion_dist_threshold)

        # Build clusters mapping: label -> list of detection indices
        clusters = defaultdict(list)
        for i, lbl in enumerate(labels):
            clusters[lbl].append(i)

        # Initialize current clusters for tracklet voting
        current = []
        for lbl, inds in clusters.items():
            cx = frame_reid[inds[0]]['x3d']
            cy = frame_reid[inds[0]]['y3d']
            # initial global_id = first detection's local_id
            gi = frame_reid[inds[0]]['local_id']
            current.append({'label': lbl, 'global_id': gi, 'center': (cx, cy)})

        # Tracklet-based global ID update (majority voting)
        for c in current:
            best_tid, best_dist = None, float('inf')
            for tid, td in active_tracklets.items():
                if frame_id - td['last_frame'] > tracklet_inactive_threshold:
                    continue
                d = np.linalg.norm(np.array(c['center']) - np.array(td['last_center']))
                if d < best_dist and d < min_obj_dis_threshold:
                    best_dist, best_tid = d, tid
            if best_tid is not None:
                td = active_tracklets[best_tid]
                td['votes'][frame_id] = c['global_id']
                td['last_center'] = c['center']
                td['last_frame'] = frame_id
                # compute majority vote in window
                window = [v for f, v in td['votes'].items() if abs(f - frame_id) <= tracklet_inactive_threshold]
                maj = Counter(window).most_common(1)[0][0]
                c['global_id'] = maj
                td['global_id'] = maj
            else:
                tracklet_counter += 1
                active_tracklets[tracklet_counter] = {
                    'global_id': c['global_id'],
                    'last_center': c['center'],
                    'last_frame': frame_id,
                    'votes': {frame_id: c['global_id']}
                }

        # Build detection-level global ID list
        det_global_ids = [None] * len(all_objs)
        for c in current:
            for idx in clusters[c['label']]:
                det_global_ids[idx] = c['global_id']

        # Annotate each frame with bounding boxes and IDs
        for i, det in enumerate(all_objs):
            cid = det['camera_id']
            img = frames[cid]
            x1, y1, x2, y2 = map(int, [det['x1'], det['y1'], det['x2'], det['y2']])
            lid = det['assigned_id']
            gid = det_global_ids[i]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"L{lid} G{gid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        # Create mosaic and stream as JPEG
        mosaic = mosaic_frames(frames)
        ret, jpeg = cv2.imencode('.jpg', mosaic)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
    # Release resources
    for cap in caps.values():
        cap.release()
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)

