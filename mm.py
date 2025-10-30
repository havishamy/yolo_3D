import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# --------------------
# 参数
# --------------------
MEMORY_TTL = 20
IOU_THRESH = 0.3

K = np.array([[604.0, 0, 334.7],
              [0, 603.7, 250.7],
              [0,   0,   1]])

# --------------------
# 工具函数：四元数 -> 旋转矩阵
# --------------------
def quat_to_rotmat(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]
    ])
    return R

# --------------------
# 读取相机位姿文件
# --------------------
def load_camera_poses(path):
    poses = {}
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if lines[0].startswith("timestamp"):
        lines = lines[1:]
    for i, line in enumerate(lines):
        ts, x, y, z, qx, qy, qz, qw = line.split(",")
        R_wc = quat_to_rotmat(float(qx), float(qy), float(qz), float(qw))
        t_wc = np.array([float(x), float(y), float(z)])
        poses[i+1] = (R_wc, t_wc)  # 第 i+1 帧对应
    return poses

# --------------------
# 初始化
# --------------------
model = YOLO("/home/dsj/code/ultralytics/ultralytics/runs/train/yolov11_glassware/weights/best.pt")
cap = cv2.VideoCapture("/home/dsj/code/ultralytics/apply/2.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("/home/dsj/code/ultralytics/apply/hh.mp4", fourcc,
                      cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# 载入相机位姿
camera_poses_all = load_camera_poses("/home/dsj/code/ultralytics/apply/2.txt")

trajectory = []   # 保存物体三维点
camera_traj = []  # 保存相机位置
frame_id = 0
prev_detections = {}  # {tid: (cx, cy)}

# --------------------
# 主循环
# --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    # 当前帧相机位姿
    if frame_id not in camera_poses_all:
        continue
    R_wc, t_wc = camera_poses_all[frame_id]
    camera_traj.append(t_wc)
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    P_curr = K @ np.hstack((R_cw, t_cw.reshape(3,1)))

    # YOLO 检测
    results = model(frame, verbose=False)[0]
    if results.boxes is not None:
        bboxes = results.boxes.xyxy.cpu().numpy()
        confs  = results.boxes.conf.cpu().numpy()
    else:
        bboxes, confs = [], []

    # 只取最高置信度目标做示例
    if len(bboxes) > 0:
        idx = np.argmax(confs)
        x1, y1, x2, y2 = bboxes[idx]
        cx, cy = (x1+x2)/2, (y1+y2)/2

        # 如果有上一帧检测
        if (frame_id-1) in prev_detections and (frame_id-1) in camera_poses_all:
            cx1, cy1 = prev_detections[frame_id-1]
            R_wc_prev, t_wc_prev = camera_poses_all[frame_id-1]
            R_cw_prev = R_wc_prev.T
            t_cw_prev = -R_cw_prev @ t_wc_prev
            P_prev = K @ np.hstack((R_cw_prev, t_cw_prev.reshape(3,1)))

            # 三角化
            pts4d = cv2.triangulatePoints(P_prev, P_curr,
                                          np.array([[cx1],[cx]]),
                                          np.array([[cy1],[cy]]))
            pts3d = (pts4d/pts4d[3])[:3].T
            trajectory.append(pts3d.squeeze())

        prev_detections[frame_id] = (cx, cy)

        # 画框
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.circle(frame, (int(cx), int(cy)), 4, (0,0,255), -1)

    out.write(frame)

cap.release()
out.release()

trajectory = np.array(trajectory)
camera_traj = np.array(camera_traj)

np.savez("/home/dsj/code/ultralytics/apply/trajectories_from_file.npz",
         object_traj=trajectory, camera_traj=camera_traj)

print("✅ 完成，结果已保存 trajectories_from_file.npz 和视频 hh.mp4")

# --------------------
# 可视化
# --------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

if len(camera_traj) > 0:
    ax.plot(camera_traj[:,0], camera_traj[:,1], camera_traj[:,2], 'b-', label="Camera Traj")
if len(trajectory) > 0:
    ax.scatter(trajectory[:,0], trajectory[:,1], trajectory[:,2], c='r', s=10, label="Object Points")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()

