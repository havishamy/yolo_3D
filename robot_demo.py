#!/usr/bin/env python3
"""
compute_3d_from_tracking.py

输入:
  - 视频文件 (VIDEO_PATH)
  - 相机位姿文件 (POSE_CSV_PATH)，格式每行:
      timestamp,cam_x,cam_y,cam_z,cam_rx,cam_ry,cam_rz,cam_rw
    这里假设文件中每一行对应视频的一帧（按顺序）。如果不是，请调整读取逻辑以按 timestamp 对齐帧。
  - YOLO 模型权重路径 MODEL_PATH

输出:
  - tracking_output_with_3d_projected.mp4  (可视化视频)
  - estimates_3d.png  (3D 散点图)
  - estimates_3d.npy  (估计数据)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# --------------------
# 用户配置（修改下面路径与参数）
# --------------------
VIDEO_PATH = "/home/dsj/code/ultralytics/apply/2.mp4"
POSE_CSV_PATH = "/home/dsj/code/ultralytics/apply/2.txt"  # 你的 pose 文件
MODEL_PATH = "/home/dsj/code/ultralytics/apply/ultralytics/runs/train/yolov11_glassware/weights/best.pt"

OUT_VIDEO = "tracking_output_with_3d_projected.mp4"
OUT_PLOT = "estimates_3d.png"
OUT_NPY = "estimates_3d.npy"


# 摄像机内参 (请按你真实相机改)
K = np.array([[604.0,   0.0, 334.7],
              [  0.0, 603.7, 250.7],
              [  0.0,   0.0,   1.0]])

IOU_THRESH = 0.3
CONF_THRESH = 0.4
MEMORY_TTL = 5

# --------------------
# 辅助函数
# --------------------
def quat_to_rotmat(qx, qy, qz, qw):
    """四元数 (x,y,z,w) -> 3x3 旋转矩阵 (world_R_camera, 即 R_wc)"""
    # normalize
    q = np.array([qx, qy, qz, qw], dtype=float)
    q = q / (np.linalg.norm(q) + 1e-16)
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    return R

def iou(b1, b2):
    x1,y1,x2,y2 = b1
    x1p,y1p,x2p,y2p = b2
    ix1 = max(x1, x1p); iy1 = max(y1, y1p)
    ix2 = min(x2, x2p); iy2 = min(y2, y2p)
    iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
    inter = iw*ih
    a1 = max(0, x2-x1)*max(0, y2-y1)
    a2 = max(0, x2p-x1p)*max(0, y2p-y1p)
    union = a1 + a2 - inter
    return inter/union if union>0 else 0.0

def ray_intersection_least_squares(origins, directions):
    """多条射线最小二乘交点，origins: list (3,), directions: list (3,)"""
    if len(origins) == 0:
        return None, None
    A = np.zeros((3,3))
    b = np.zeros(3)
    for o,d in zip(origins,directions):
        d = d / (np.linalg.norm(d)+1e-12)
        Ai = np.eye(3) - np.outer(d,d)
        A += Ai
        b += Ai @ o
    # 求解
    try:
        X = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(A, b, rcond=None)[0]
    # residual
    dists = []
    for o,d in zip(origins,directions):
        d = d / (np.linalg.norm(d)+1e-12)
        proj = o + d * (d @ (X - o))
        dists.append(np.linalg.norm(proj - X))
    rms = float(np.sqrt(np.mean(np.array(dists)**2))) if len(dists)>0 else 0.0
    return X, rms

def project_point_world_to_image(K, R_wc, t_wc, X_world):
    """
    把世界点投影到图像：
      X_cam = R_cw @ (X_world - t_wc)  , R_cw = R_wc^T
      uv = K @ X_cam
    返回 (u,v,depth) 或 None(若深度<=0)
    """
    R_cw = R_wc.T
    X_cam = R_cw @ (X_world - t_wc.reshape(3,))
    if X_cam[2] <= 1e-6:
        return None
    uv = K @ X_cam
    uv = uv / uv[2]
    return int(round(uv[0])), int(round(uv[1])), float(X_cam[2])

# --------------------
# 简单 Kalman 跟踪（基于 bbox 中心），借用你原实现的简化版
# --------------------
class BBoxKalman:
    def __init__(self, x1,y1,x2,y2, max_jump=40):
        cx,cy = (x1+x2)/2.0, (y1+y2)/2.0
        w,h = (x2-x1),(y2-y1)
        self.x = np.array([cx,cy,w,h,0,0,0,0], dtype=float)
        self.P = np.eye(8) * 10.0
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i+4] = 1.0
        self.H = np.zeros((4,8)); self.H[0,0]=1; self.H[1,1]=1; self.H[2,2]=1; self.H[3,3]=1
        self.R = np.eye(4) * 5.0
        self.Q = np.eye(8) * 0.01
        self.max_jump = max_jump
        self.last_pos = (cx,cy,w,h)
    def step(self, x1=None,y1=None,x2=None,y2=None):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if x1 is not None:
            cx,cy = (x1+x2)/2.0, (y1+y2)/2.0
            w,h = (x2-x1),(y2-y1)
            prev = self.last_pos
            d = np.linalg.norm([cx-prev[0], cy-prev[1]])
            if d < self.max_jump:
                z = np.array([cx,cy,w,h])
                y = z - (self.H @ self.x)
                S = self.H @ self.P @ self.H.T + self.R
                K = self.P @ self.H.T @ np.linalg.inv(S)
                self.x = self.x + K @ y
                self.P = (np.eye(8) - K @ self.H) @ self.P
        out_cx,out_cy,out_w,out_h = self.x[0], self.x[1], self.x[2], self.x[3]
        x1 = int(out_cx - out_w/2.0); y1 = int(out_cy - out_h/2.0)
        x2 = int(out_cx + out_w/2.0); y2 = int(out_cy + out_h/2.0)
        self.last_pos = (out_cx,out_cy,out_w,out_h)
        return (x1,y1,x2,y2), (out_cx,out_cy)

# --------------------
# 读入相机位姿文件
# --------------------
def load_camera_poses(pose_path):
    """
    返回字典: frame_idx (从1开始) -> (R_wc (3x3), t_wc (3,))
    假设文件每行对应一帧（按顺序）。如果你的 csv 第一行为 header，会自动跳过。
    """
    poses = {}
    if not os.path.exists(pose_path):
        raise FileNotFoundError(pose_path)
    with open(pose_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # skip header if present
    if len(lines)>0 and lines[0].lower().startswith("timestamp"):
        lines = lines[1:]
    for i,ln in enumerate(lines):
        parts = ln.split(',')
        if len(parts) < 8:
            continue
        # parse
        ts = float(parts[0])
        cx = float(parts[1]); cy = float(parts[2]); cz = float(parts[3])
        qx = float(parts[4]); qy = float(parts[5]); qz = float(parts[6]); qw = float(parts[7])
        R_wc = quat_to_rotmat(qx,qy,qz,qw)  # world_R_camera
        t_wc = np.array([cx,cy,cz], dtype=float)
        frame_idx = i+1  # assume first line -> frame 1
        poses[frame_idx] = (R_wc, t_wc, ts)
    return poses

# --------------------
# 主流程
# --------------------
def main():
    # load camera poses
    cam_poses = load_camera_poses(POSE_CSV_PATH)
    print(f"[INFO] loaded {len(cam_poses)} camera poses from {POSE_CSV_PATH}")

    # load model & video
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w,h))
    print(f"[INFO] video opened, fps={fps}, size=({w},{h})")

    # trackers
    trackers = {}   # id -> dict
    next_id = 0
    frame_idx = 0

    # store per-frame detections and associated frame camera pose (if exists)
    detections_per_frame = {}   # frame_idx -> list of {id,bbox,conf,cls}
    camera_pose_per_frame = {}  # frame_idx -> (R_wc, t_wc, ts)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # record camera pose if available for this frame index
        if frame_idx in cam_poses:
            camera_pose_per_frame[frame_idx] = cam_poses[frame_idx][:2]  # R_wc, t_wc
        # else: no pose for this frame

        img = frame.copy()
        # run detection
        results = model(img, verbose=False)[0]
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()
            # filter by conf
            mask = confs >= CONF_THRESH
            boxes = boxes[mask]; confs = confs[mask]; clss = clss[mask]
        else:
            boxes = np.zeros((0,4)); confs = np.zeros((0,)); clss = np.zeros((0,))

        # mark existing trackers as aged
        for tid, d in list(trackers.items()):
            d['ttl'] -= 1

        matched_ids = set()
        # associate by IoU
        for bbox, conf, cls in zip(boxes, confs, clss):
            x1,y1,x2,y2 = bbox
            best_id = None; best_iou = 0.0
            for tid, d in trackers.items():
                i = iou(d['bbox'], (x1,y1,x2,y2))
                if i>IOU_THRESH and i>best_iou:
                    best_iou = i; best_id = tid
            if best_id is not None:
                # update
                kf = trackers[best_id]['kf']
                sm_bbox, center = kf.step(x1,y1,x2,y2)
                trackers[best_id]['bbox'] = sm_bbox
                trackers[best_id]['ttl'] = MEMORY_TTL
                trackers[best_id]['conf'] = float(conf)
                trackers[best_id]['cls'] = int(cls)
                matched_ids.add(best_id)
            else:
                # create
                kf = BBoxKalman(x1,y1,x2,y2)
                trackers[next_id] = {'kf': kf,
                                     'bbox': np.array([int(x1),int(y1),int(x2),int(y2)]),
                                     'ttl': MEMORY_TTL,
                                     'conf': float(conf),
                                     'cls': int(cls),
                                     'trajectory': []}
                matched_ids.add(next_id)
                next_id += 1

        # remove expired
        expired = [tid for tid,d in trackers.items() if d['ttl']<=0]
        for tid in expired:
            del trackers[tid]

        # record detections for this frame
        detections_per_frame[frame_idx] = []
        for tid, d in trackers.items():
            x1,y1,x2,y2 = map(int, d['bbox'])
            detections_per_frame[frame_idx].append({
                'id': tid, 'bbox': [x1,y1,x2,y2], 'conf': d['conf'], 'cls': d['cls']
            })
            # draw on frame
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cx = int((x1+x2)/2); cy = int((y1+y2)/2)
            cv2.circle(img, (cx,cy), 3, (0,0,255), -1)
            cv2.putText(img, f"ID{tid} {d['conf']:.2f}", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        writer.write(img)

    cap.release()
    writer.release()
    print(f"[INFO] detection+tracking finished on {frame_idx} frames. Saved temp video {OUT_VIDEO}")

    # --------------------
    # 通过射线求交估计每个目标的 3D 位置
    # --------------------
    estimates = {}
    for tid in range(next_id):
        origins = []
        directions = []
        obs_frames = []
        for fid, dets in detections_per_frame.items():
            for det in dets:
                if det['id'] != tid:
                    continue
                # need camera pose available for this frame
                if fid not in camera_pose_per_frame:
                    continue
                R_wc, t_wc = camera_pose_per_frame[fid]
                x1,y1,x2,y2 = det['bbox']
                u = (x1 + x2) / 2.0
                v = (y1 + y2) / 2.0
                # direction in cam coords
                vec_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
                # convert to world direction: d_world = R_wc @ vec_cam
                d_world = R_wc @ vec_cam
                origins.append(t_wc.astype(float))
                directions.append(d_world.astype(float))
                obs_frames.append(fid)
        if len(origins) >= 2:
            X, rms = ray_intersection_least_squares(origins, directions)
            estimates[tid] = {'point': X, 'rms': rms, 'num_obs': len(origins), 'obs_frames': obs_frames}
        else:
            estimates[tid] = {'point': None, 'rms': None, 'num_obs': len(origins), 'obs_frames': obs_frames}

    # save estimates
    np.save(OUT_NPY, estimates)
    print(f"[INFO] estimates saved to {OUT_NPY}")

    # print summary
    print("=== estimates summary ===")
    for tid, info in estimates.items():
        print(f"ID {tid}: num_obs={info['num_obs']}, rms={info['rms']}, point={info['point']}")

    # --------------------
    # 可视化 1: 3D plot (camera centers + estimated points)
    # --------------------
    camera_centers = []
    # sort frame ids
    frames_sorted = sorted(camera_pose_per_frame.keys())
    for fid in frames_sorted:
        R_wc, t_wc = camera_pose_per_frame[fid]
        camera_centers.append(t_wc.reshape(3,))
    camera_centers = np.array(camera_centers) if len(camera_centers)>0 else np.zeros((0,3))

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    if camera_centers.shape[0] > 0:
        ax.plot(camera_centers[:,0], camera_centers[:,1], camera_centers[:,2], '-o', label='camera centers')
    for tid, info in estimates.items():
        if info['point'] is not None:
            X = info['point']
            ax.scatter([X[0]], [X[1]], [X[2]], s=60, label=f"id{tid} n={info['num_obs']}")
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=200)
    print(f"[INFO] 3D plot saved to {OUT_PLOT}")

    # --------------------
    # 可视化 2: 将估计点投影回视频并保存
    # --------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    writer2 = cv2.VideoWriter("tracking_output_with_3d_projected.mp4", fourcc, fps, (w,h))
    fid = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fid += 1
        # draw projected estimates if this frame has camera pose
        if fid in camera_pose_per_frame:
            R_wc, t_wc = camera_pose_per_frame[fid]
            for tid, info in estimates.items():
                if info['point'] is None:
                    continue
                X = info['point']
                proj = project_point_world_to_image(K, R_wc, t_wc, X)
                if proj is None:
                    continue
                u,v,depth = proj
                # clamp
                if 0 <= u < w and 0 <= v < h:
                    cv2.circle(frame, (u,v), 6, (0,0,255), -1)
                    cv2.putText(frame, f"ID{tid} d={depth:.3f}", (u+6, v-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        writer2.write(frame)
    cap.release()
    writer2.release()
    print("[INFO] projection video saved to tracking_output_with_3d_projected.mp4")
    print("[DONE] All outputs saved.")

if __name__ == "__main__":
    main()
