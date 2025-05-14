import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
import cv2

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

def load_camera_parameters(calib_file, camera_index):
    """
    Đọc K, R, C của MỘT camera từ file calibration.xml.

    Trả về: K, R, C cho camera cụ thể
    """
    fs = cv2.FileStorage(calib_file, cv2.FileStorage_READ)
    if not fs.isOpened():
        raise IOError(f"Cannot open calibration file: {calib_file}")

    # Đọc node K{i}, R{i}, C{i}
    K_node = fs.getNode(f"K{camera_index}")
    R_node = fs.getNode(f"R{camera_index}")
    C_node = fs.getNode(f"C{camera_index}")

    # Kiểm tra node
    if K_node.empty() or R_node.empty() or C_node.empty():
        raise ValueError(f"Không tìm thấy K{camera_index}, R{camera_index} hoặc C{camera_index} trong file XML")

    # Lấy ma trận
    K = K_node.mat()  # 3x3
    R = R_node.mat()  # 3x3
    C = C_node.mat()  # 3x1

    # Chuyển vector C từ shape (3,1) sang (3,)
    C = C.reshape((3,))

    fs.release()
    return K, R, C


def pixel_to_world_ground(u, v, K, R, C):
    """
    Chuyển pixel (u,v) thành (X,Y,0) trong hệ world, giả định điểm nằm trên mặt phẳng Z=0.
    - K: ma trận nội (3x3)
    - R: ma trận quay (3x3) từ world->camera
    - C: toạ độ camera trong world (3,)

    Trả về (X, Y, Z) = (X, Y, 0)
    """
    # 1) Vector ảnh trong toạ độ camera chuẩn hoá
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    invK = np.linalg.inv(K)
    d_c = invK @ uv1  # hướng tia trong hệ camera

    # 2) Đổi sang hướng tia trong hệ world
    #    (X_w = C + t * R^T * X_c)
    d_w = R.T @ d_c  # vector 3D trong world

    # 3) Tìm t để Z=0
    cz = C[2]
    dz = d_w[2]
    if abs(dz) < 1e-12:
        # Trường hợp camera nằm đúng mặt phẳng Z=0 hoặc tia song song
        return None  # hoặc raise Exception(...)
    t = -cz / dz

    # 4) Tính toạ độ giao
    X = C[0] + t * d_w[0]
    Y = C[1] + t * d_w[1]
    Z = 0.0  # giả thiết mặt phẳng Z=0

    return (X, Y, Z)


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.ground_3d = None  # Thêm thuộc tính để lưu tọa độ 3D ground

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30, calib_file=None, camera_index=None):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * 30)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # Load camera parameters
        self.calib_file = calib_file
        self.camera_index = camera_index
        if calib_file is not None and camera_index is not None:
            self.K, self.R, self.C = load_camera_parameters(calib_file, camera_index)
        else:
            self.K, self.R, self.C = None, None, None

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []
        print(f"Frame {self.frame_id}: Num of detections = {len(detections)}")
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                # Tính tọa độ 3D ground
                if self.K is not None and self.R is not None and self.C is not None:
                    tlbr = track.tlbr
                    x1, y1, x2, y2 = tlbr
                    cx = (x1 + x2) / 2
                    cy = y2
                    ground_3d = pixel_to_world_ground(cx, cy, self.K, self.R, self.C)
                    if ground_3d is not None:
                        track.ground_3d = ground_3d
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                # Tính tọa độ 3D ground
                if self.K is not None and self.R is not None and self.C is not None:
                    tlbr = track.tlbr
                    x1, y1, x2, y2 = tlbr
                    cx = (x1 + x2) / 2
                    cy = y2
                    ground_3d = pixel_to_world_ground(cx, cy, self.K, self.R, self.C)
                    if ground_3d is not None:
                        track.ground_3d = ground_3d
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                # Tính tọa độ 3D ground
                if self.K is not None and self.R is not None and self.C is not None:
                    tlbr = track.tlbr
                    x1, y1, x2, y2 = tlbr
                    cx = (x1 + x2) / 2
                    cy = y2
                    ground_3d = pixel_to_world_ground(cx, cy, self.K, self.R, self.C)
                    if ground_3d is not None:
                        track.ground_3d = ground_3d
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                # Tính tọa độ 3D ground
                if self.K is not None and self.R is not None and self.C is not None:
                    tlbr = track.tlbr
                    x1, y1, x2, y2 = tlbr
                    cx = (x1 + x2) / 2
                    cy = y2
                    ground_3d = pixel_to_world_ground(cx, cy, self.K, self.R, self.C)
                    if ground_3d is not None:
                        track.ground_3d = ground_3d
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            # Tính tọa độ 3D ground
            if self.K is not None and self.R is not None and self.C is not None:
                tlbr = unconfirmed[itracked].tlbr
                x1, y1, x2, y2 = tlbr
                cx = (x1 + x2) / 2
                cy = y2
                ground_3d = pixel_to_world_ground(cx, cy, self.K, self.R, self.C)
                if ground_3d is not None:
                    unconfirmed[itracked].ground_3d = ground_3d
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            # Tính tọa độ 3D ground
            if self.K is not None and self.R is not None and self.C is not None:
                tlbr = track.tlbr
                x1, y1, x2, y2 = tlbr
                cx = (x1 + x2) / 2
                cy = y2
                ground_3d = pixel_to_world_ground(cx, cy, self.K, self.R, self.C)
                if ground_3d is not None:
                    track.ground_3d = ground_3d
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
