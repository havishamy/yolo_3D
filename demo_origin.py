import cv2
import numpy as np
from ultralytics import YOLO
#from filterpy.kalman import KalmanFilter
import json

# --------------------
# 参数
# --------------------
MEMORY_TTL = 20   # 丢失多少帧内继续保留目标
IOU_THRESH = 0.3  # IoU 阈值用于目标关联

K = np.array([[604.0, 0, 334.7],
              [0, 603.7, 250.7],
              [0,   0,   1]])

class BBoxKalman:
    def __init__(self, x1, y1, x2, y2, max_jump=20):
        # 初始中心和宽高
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = (x2 - x1), (y2 - y1)

        # 状态 [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
        self.P = np.eye(8) * 10.0

        # 状态转移矩阵
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i+4] = 1  # 位置受速度影响

        # 观测矩阵
        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1  # cx
        self.H[1, 1] = 1  # cy
        self.H[2, 2] = 1  # w
        self.H[3, 3] = 1  # h

        self.R = np.eye(4) * 5.0
        self.Q = np.eye(8) * 0.01

        self.max_jump = max_jump
        self.last_pos = (cx, cy, w, h)

    def step(self, x1=None, y1=None, x2=None, y2=None):
        """
        执行一步预测+更新
        输入: 检测框 (x1, y1, x2, y2)，None 表示检测失败
        返回: 平滑后的检测框 (x1, y1, x2, y2), (cx, cy)
        """
        # ---------- 预测 ----------
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        pred_cx, pred_cy, pred_w, pred_h = self.x[0], self.x[1], self.x[2], self.x[3]

        # ---------- 更新 ----------
        if x1 is not None:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = (x2 - x1), (y2 - y1)

            prev_cx, prev_cy, prev_w, prev_h = self.last_pos
            d = np.linalg.norm([cx - prev_cx, cy - prev_cy])

            if d < self.max_jump:  # 合理 → 更新
                z = np.array([cx, cy, w, h])
                y = z - self.H @ self.x
                S = self.H @ self.P @ self.H.T + self.R
                K = self.P @ self.H.T @ np.linalg.inv(S)
                self.x = self.x + K @ y
                self.P = (np.eye(8) - K @ self.H) @ self.P

        # ---------- 取输出 ----------
        smooth_cx, smooth_cy, smooth_w, smooth_h = self.x[0], self.x[1], self.x[2], self.x[3]
        smooth_x1 = int(smooth_cx - smooth_w / 2)
        smooth_y1 = int(smooth_cy - smooth_h / 2)
        smooth_x2 = int(smooth_cx + smooth_w / 2)
        smooth_y2 = int(smooth_cy + smooth_h / 2)

        self.last_pos = (smooth_cx, smooth_cy, smooth_w, smooth_h)
        return (smooth_x1, smooth_y1, smooth_x2, smooth_y2),(smooth_cx, smooth_cy)


#----------------
# IoU 匹配函数
# --------------------
def iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_p, y1_p, x2_p, y2_p = bbox2
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0

# --------------------
# 初始化
# --------------------
model = YOLO("/home/dsj/code/ultralytics/ultralytics/runs/train/yolov11_glassware/weights/best.pt")  # 替换成你的模型路径
cap = cv2.VideoCapture("/home/dsj/code/ultralytics/apply/input_2.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("/home/dsj/code/ultralytics/apply/tracking_output_karman_2.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

orb = cv2.ORB_create(1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

R_total = np.eye(3)
t_total = np.zeros((3, 1))

R_prev = R_total.copy()
t_prev = t_total.copy()

camera_traj = []     # 相机轨迹位置 (N, 3)
camera_poses = []    # 相机位姿 (N, 3, 4) 

trajectory = []
# 跟踪器存储 {id: {...}}
trackers = {}
next_id = 0
#results_all = {}

frame_id = 0

# --------------------
# 主循环
# --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # YOLO 检测
    results = model(frame, verbose=False)[0]
    bboxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
    confs  = results.boxes.conf.cpu().numpy() if results.boxes is not None else []
    clss   = results.boxes.cls.cpu().numpy() if results.boxes is not None else []

    matched_ids = set()
  #  frame_objects = []

    # 先更新已有 track 的预测
    for tid, data in trackers.items():
        data["ttl"] -= 1

    # 检测结果与已有 track 关联
    for bbox, conf, cls in zip(bboxes, confs, clss):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        best_iou = 0
        best_id = None
        for tid, data in trackers.items():
            iou_score = iou(bbox, data["bbox"])
            if iou_score > best_iou and iou_score > IOU_THRESH:
                best_iou = iou_score
                best_id = tid

        if best_id is not None:
            # 更新已有 track
            trackers[best_id]["bbox"]=trackers[best_id]["kf"].step(x1, y1, x2, y2)[0]
            #trackers[best_id]["bbox"] = bbox
            trackers[best_id]["ttl"] = MEMORY_TTL
            trackers[best_id]["conf"] = float(conf)
            trackers[best_id]["cls"] = int(cls)
            matched_ids.add(best_id)
        else:
            # 新建 track
            kf = BBoxKalman(x1, y1, x2, y2)
            trackers[next_id] = {
                "kf": kf,
                "bbox": bbox,
                "ttl": MEMORY_TTL,
                "conf": float(conf),
                "cls": int(cls),
                "prev_kp":None,
                "kp":None,
                "prev_des":None,
                "des":None
            }
            matched_ids.add(next_id)
            next_id += 1

    # 删除过期 track
    expired = [tid for tid, data in trackers.items() if data["ttl"] <= 0]
    for tid in expired:
        del trackers[tid]

    for tid, data in trackers.items():
        x1, y1, x2, y2 = map(int, data["bbox"])
        cx, cy ,w,h= data["kf"].x[0],data["kf"].x[1],data["kf"].x[2],data["kf"].x[3]

        # 在 ROI 内提取 ORB
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[int(cy-h//4):int(cy+h//4), int(cx-w//4):int(cx+w//4)] = 255
        kp, des = orb.detectAndCompute(gray, mask)
        data["kp"], data["des"] = kp, des

        # 如果有上一帧 → 匹配
        if data["prev_kp"] is not None and data["prev_des"] is not None and data["des"] is not None:
            matches = bf.match(data["prev_des"], data["des"])
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > 8:
                pts1 = np.float32([data["prev_kp"][m.queryIdx].pt for m in matches]).reshape(-1, 2)
                pts2 = np.float32([data["kp"][m.trainIdx].pt for m in matches]).reshape(-1, 2)
                pts1 = np.vstack([pts1, [data["prev_center"][0], data["prev_center"][1]]])
                pts2 = np.vstack([pts2, [cx, cy]])

                # 估计相机运动
                E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None:
                    _, R, t, mask_pose = cv2.recoverPose(E, pts2, pts1, K)
                    t_total += R_total @ t
                    R_total = R @ R_total
                    camera_traj.append(t_total.ravel().copy())

                    # 保存相机姿态 (3x4 矩阵)
                    pose = np.hstack((R_total, t_total))
                    camera_poses.append(pose)

                    # 三角测量中心点
                    cx1, cy1 = data["prev_center"] if data["prev_center"] else (cx, cy)
                    cx2, cy2 = cx, cy
                    P1 = K @ np.hstack((R_prev, t_prev))
                    P2 = K @ np.hstack((R_total, t_total))
                    pts4d = cv2.triangulatePoints(
                        P1, P2,
                        np.array([[cx1], [cy1]]),
                        np.array([[cx2], [cy2]])
                    )
                    pts3d = (pts4d / pts4d[3])[:3].T
                    trajectory.append(pts3d)

                    R_prev = R_total.copy()
                    t_prev = t_total.copy()
        data["prev_kp"] = data.get("kp", None)
        data["prev_des"] = data.get("des", None)
        data["prev_center"] = (cx, cy)
    # 保存结果 + 画到视频
        '''
        # 结果保存
        frame_objects.append({
            "id": tid,
            "bbox": data["bbox"].tolist(),
            "center": [float(cx), float(cy)],
            "cls": data["cls"],
            "conf": data["conf"]
        })
        '''
        # 画到视频
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        for kp_ in kp:
            x, y = map(int, kp_.pt)
            cv2.circle(frame, (x,y), 2, (255,0,0), -1) 
        cv2.putText(frame, f"ID {tid} Cls {data['cls']} {data['conf']:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)

    #results_all[frame_id] = frame_objects
    out.write(frame)

cap.release()
out.release()

camera_traj = np.array(camera_traj)      # (N, 3)
camera_poses = np.array(camera_poses)    # (N, 3, 4)
object_traj = np.array(trajectory)      # (N, 3)

np.savez("/home/dsj/code/ultralytics/apply/trajectories_2.npz", 
         camera_traj=camera_traj, 
         camera_poses=camera_poses, 
         object_traj=object_traj)
# 保存 JSON
'''
with open("tracking_results.json", "w") as f:
    json.dump(results_all, f, indent=2)
'''
print(f"✅ 视频已保存: tracking_output.mp4")
#print(f"✅ 结果已保存: tracking_results.json (帧数={len(results_all)})")
