"""
multi_view_yolo_triangulate.py

功能：
 - 使用 YOLOv8 做检测 + 跟踪（tracker="bytetrack.yaml"）
 - 为每个 track 收集多帧的 bbox center (u,v) 与对应相机在 base 下位姿 T_base_cam
 - 对每个 track 做鲁棒三角化，得到 3D 中心（base 坐标系）
 - 支持两种获取相机位姿方式：ROS TF（实时）或离线位姿列表文件

用法示例：
  python multi_view_yolo_triangulate.py --video input.mp4 --model yolov8n.pt --mode offline \
      --poses poses.npy --Kfx 615 --Kfy 615 --Kcx 320 --Kcy 240

或者在 ROS 环境中：
  python multi_view_yolo_triangulate.py --video input.mp4 --model yolov8n.pt --mode ros \
      --camera_frame camera_color_optical_frame --base_frame base_link --Kfx ... 
"""

import argparse
import time
import csv
from collections import defaultdict, deque
from typing import List, Tuple, Dict
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


# YOLOv8
from ultralytics import YOLO

# Optional ROS imports (only used if mode == 'ros')
HAS_ROS = False
try:
    import rospy
    import tf2_ros
    import geometry_msgs.msg
    HAS_ROS = True
except Exception:
    HAS_ROS = False

def show_ray(points,directions):
    points=np.array(points)
    directions=np.array(directions)
    # 创建3D图像
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', s=1, label='Points')

    # 绘制方向向量（箭头）
    for p, d in zip(points, directions):
        ax.quiver(p[0], p[1], p[2], d[0], d[1], d[2],
                length=1.0, normalize=True, color='b',linewidth=0.1,arrow_length_ratio=0.2)

    # 设置坐标轴范围与标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 2)
    ax.set_zlim(-4, -1)
    ax.set_title("3D Points and Direction Vectors")

    # 图例与显示
    ax.legend()
    plt.tight_layout()

    # 保存为图片
    plt.savefig('3d_vectors.png', dpi=300)

# ---------------------
# Helper math functions
# ---------------------
def draw_axes(ax, T, length=0.05, lw=1):
    """在三维图上画一个小坐标系，表示相机或机械臂末端的姿态"""
    origin = T[:3, 3]
    R = T[:3, :3]
    x_axis = origin + R[:, 0] * length
    y_axis = origin + R[:, 1] * length
    z_axis = origin + R[:, 2] * length
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 'r', lw=lw)
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 'g', lw=lw)
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 'b', lw=lw)
    
    
def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def pixel_to_ray(u: float, v: float, K: np.ndarray) -> np.ndarray:
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    x_cam = np.linalg.inv(K).dot(uv1)
    d = x_cam / np.linalg.norm(x_cam)
    return d

def skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=np.float64)

def triangulate_rays(cam_centers: List[np.ndarray], directions: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    Solve for X in base frame minimizing perpendicular distances to rays.
    Returns X (3,), mean_perp_distance
    """
    assert len(cam_centers) == len(directions) and len(cam_centers) >= 2
    show_ray(cam_centers,directions)
    A_blocks = []
    b_blocks = []
    for C, d in zip(cam_centers, directions):
        d = d / np.linalg.norm(d)
        S = skew(d)
        A_blocks.append(S)
        b_blocks.append(S.dot(C))
    A = np.vstack(A_blocks)
    b = np.hstack(b_blocks)
    X, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    # compute mean perp dist
    dists = []
    for C, d in zip(cam_centers, directions):
        v = X - C
        proj = np.dot(v, d) * d
        perp = v - proj
        dists.append(np.linalg.norm(perp))
    return X, float(np.mean(dists))

def robust_triangulate(cam_centers, directions, min_views=2, max_iter=5, thresh=0.05):
    """
    Iterative outlier removal:
     - solve with all rays
     - compute perp distances; remove rays with dist > thresh
     - repeat until stable or less than min_views left
    returns X, mean_error, used_indices
    """
    indices = list(range(len(cam_centers)))
    for it in range(max_iter):
        if len(indices) < min_views:
            break
        C_sel = [cam_centers[i] for i in indices]
        d_sel = [directions[i] for i in indices]
        X, mean_err = triangulate_rays(C_sel, d_sel)
        # compute perp distances
        perp = []
        for Ci, di in zip(C_sel, d_sel):
            v = X - Ci
            proj = np.dot(v, di) * di
            perp.append(np.linalg.norm(v - proj))
        perp = np.array(perp)
        # find indices to keep
        keep_mask = perp <= thresh
        if keep_mask.all():
            return X, float(perp.mean()), indices
        # otherwise remove worst offenders (those > thresh)
        new_indices = [idx for k, idx in zip(keep_mask, indices) if k]
        if len(new_indices) == len(indices):
            # none removed -> stop
            return X, float(perp.mean()), indices
        indices = new_indices
    # final solve with remaining indices (if enough)
    if len(indices) >= min_views:
        C_sel = [cam_centers[i] for i in indices]
        d_sel = [directions[i] for i in indices]
        X, mean_err = triangulate_rays(C_sel, d_sel)
        return X, mean_err, indices
    else:
        return None, None, []

# ---------------------
# Pose utilities
# ---------------------
def tf_transform_to_matrix(tf_trans):
    """
    Convert geometry_msgs/TransformStamped or Transform to 4x4 numpy matrix
    """
    t = tf_trans.transform.translation
    q = tf_trans.transform.rotation
    tx, ty, tz = t.x, t.y, t.z
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    # convert quaternion to rotation matrix (simple formula)
    R = quat_to_rot_matrix([qx, qy, qz, qw])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T

def quat_to_rot_matrix(q):
    qx, qy, qz, qw = q
    # normalize
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    # rotation matrix
    R = np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),   1-2*(qx*qx+qz*qz),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw),   1-2*(qx*qx+qy*qy)]
    ], dtype=np.float64)
    return R

# ---------------------
# Main processing class
# ---------------------
class MultiViewTriangulator:
    def __init__(self, model_path, K, video_path, mode='offline', poses_file=None,
                 base_frame='base', camera_frame='camera_color_frame',
                 min_views=3, out_csv='triangulated_results.csv',save_plot_path="triangulated_results.png"):
        """
        mode: 'offline' -> read poses from poses_file (numpy .npy or .npz or csv mapping)
              'ros' -> read poses from TF based on frame timestamps
        poses_file: if offline, should be array-like shape (N_frames,4,4) or csv with flattened 16 values per row
        """
        self.model = YOLO(model_path)
        self.K = np.asarray(K, dtype=np.float64)
        self.cap = cv2.VideoCapture(video_path)
        self.mode = mode
        self.poses_file = poses_file
        self.base_frame = base_frame
        self.camera_frame = camera_frame
        self.min_views = min_views
        self.out_csv = out_csv
        self.save_plot_path = save_plot_path

        # storage: per track_id -> list of (frame_idx, timestamp, u,v, T_base_cam)
        self.track_observations: Dict[int, List[Dict]] = defaultdict(list)

        # if offline mode load poses file
        self.poses_array = None
        if self.mode == 'offline':
            if poses_file is None:
                raise ValueError("poses_file must be provided in offline mode")
            # support npy/npz or csv
            if poses_file.endswith('.npy') or poses_file.endswith('.npz'):
                self.poses_array = np.load(poses_file)
            if poses_file.endswith('.txt'):
                self.poses_array = np.loadtxt(poses_file, delimiter=",",skiprows=1)
            else:
                # try csv: each row 16 numbers row-major
                mats = []
                with open(poses_file, 'r') as f:
                    rdr = csv.reader(f)
                    for row in rdr:
                        vals = [float(x) for x in row]
                        assert len(vals) == 16
                        mats.append(np.array(vals).reshape(4,4))
                self.poses_array = np.stack(mats, axis=0)
            poses_list = []
        for row in self.poses_array:
            row = np.asarray(row, dtype=np.float64)

            # 支持两种格式：[timestamp, tx, ty, tz, qx, qy, qz, qw] 或 [tx, ty, tz, qx, qy, qz, qw]
            if len(row) == 8:
                _, tx, ty, tz, qx, qy, qz, qw = row
            elif len(row) == 7:
                tx, ty, tz, qx, qy, qz, qw = row
            else:
                print(f"[WARN] invalid row length ({len(row)}): {row}")
                continue

            # 构建4x4齐次变换矩阵
            T = np.eye(4, dtype=np.float64)
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, :3] = rot
            T[:3, 3] = [tx, ty, tz]

            poses_list.append(T)

        self.poses_array = poses_list  # list of 4×4 matrices

        # if ros mode init tf buffer
        self.tf_buffer = None
        if self.mode == 'ros':
            if not HAS_ROS:
                raise RuntimeError("ROS mode requested but rospy/tf2_ros not available")
            rospy.init_node('multi_view_triangulator_node', anonymous=True)
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # params for robust tri
        self.tri_min_views = max(2, min_views)
        self.tri_thresh = 0.05  # meters; outlier threshold
        self.tri_max_iter = 6

    def get_pose_for_frame(self, frame_idx, frame_time_sec=None):
        """
        Return 4x4 T_base_cam for given frame index.
        If mode == offline: read from self.poses_array[frame_idx]
        If mode == ros: query tf at timestamp (use frame_time_sec)
        """
        if self.mode == 'offline':
            if frame_idx < 0 or frame_idx >= len(self.poses_array):
                return None
            T = self.poses_array[frame_idx]

            return T
        else:
            # ROS mode: query TF
            if frame_time_sec is None:
                stamp = rospy.Time.now()
            else:
                stamp = rospy.Time.from_sec(frame_time_sec)
            try:
                # lookup_transform(target_frame, source_frame, time, timeout)
                trans = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, stamp, rospy.Duration(0.5))
                T = tf_transform_to_matrix(trans)
                return T
            except Exception as e:
                print(f"[WARN] TF lookup failed for frame {frame_idx} at time {frame_time_sec}: {e}")
                return None

    def process(self, max_frames=None, save_frames_output=False):
        """
        Main loop: read video frames, run detection+tracking, collect observations.
        After processing, run triangulation for each track with enough observations.
        """
        frame_idx = -1
        results_history = None

        # We'll call model.track on the whole video using ultralytics high-level API,
        # but we want per-frame results to be able to map to poses. So we iterate frames and call model.track(frame).
        # Note: for performance you may want to use model.predict in batches and external tracker;
        # here we prefer clarity and correctness.
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[INFO] end of video")
                break
            frame_idx += 1
            t0 = time.time()

            # get frame timestamp (for ROS TF) - approximate with system time or video timestamp if available
            # OpenCV doesn't provide ros timestamps, so we use wall-time here unless offline poses align with frame index.
            frame_time_sec = time.time()

            # run detection+tracking on this single frame
            # use persist=True so boxes/ids are kept in results
            # model.track can accept numpy array as source, returns list-like results
            try:
                results = self.model.track(source=frame, show=False, persist=True, tracker="bytetrack.yaml")
            except Exception as e:
                # fallback: use predict and no tracking
                print(f"[WARN] model.track failed: {e}. Try model.predict and interpret as single-frame detections.")
                results = self.model.predict(frame)

            # results is list; get first element for this single-frame input
            if isinstance(results, list) and len(results) > 0:
                res = results[0]
            else:
                print("[WARN] Empty detection result, skipping frame", frame_idx)
                continue

            # parse boxes and track ids
            # Ultralyitcs results.boxes has attributes: xyxy, conf, cls, id (if tracking)
            boxes = []
            ids = []
            try:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, 'xyxy') else np.array([])
                # some ultralytics versions store .id
                has_id = hasattr(res.boxes, 'id')
                if has_id and res.boxes.id is not None:
                    ids_np = res.boxes.id.cpu().numpy()
                else:
                    # fallback: no tracking id -> assign incremental negative id (no triangulation)
                    ids_np = np.arange(len(boxes_xyxy)) * -1
                for b, tid in zip(boxes_xyxy, ids_np):
                    x1, y1, x2, y2 = [float(x) for x in b]
                    boxes.append((x1, y1, x2, y2))
                    ids.append(int(tid))
            except Exception as e:
                print("[ERROR] parsing results:", e)
                continue

            # get camera pose for this frame
            T_base_cam = self.get_pose_for_frame(frame_idx, frame_time_sec)  # 4x4 or None
            if T_base_cam is None:
                # we can still collect observations but triangulation will fail later; skip if pose absent
                print(f"[WARN] No pose for frame {frame_idx}, skipping observations for this frame")
                # optionally continue but here skip
                continue

            # process detections: compute centers and store by track id
            for bbox, tid in zip(boxes, ids):
                u, v = bbox_center(bbox)
                # build observation entry
                obs = {
                    'frame_idx': frame_idx,
                    'time': frame_time_sec,
                    'u': float(u),
                    'v': float(v),
                    'T_base_cam': T_base_cam  # store reference (copy)
                }
                # append to track history
                self.track_observations[tid].append(obs)

                # visualize bbox + id on frame
                x1,y1,x2,y2 = map(int, bbox)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"ID:{tid}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            '''
            # show frame
            cv2.imshow("tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] user quit")
                break
            '''

            # optional: early stop
            if max_frames is not None and frame_idx+1 >= max_frames:
                print("[INFO] reached max_frames", max_frames)
                break

        # post-processing: for each track, triangulate if enough observations
        print("[INFO] Starting triangulation for tracks...")
        results_3d = []
        for tid, obs_list in self.track_observations.items():
            if len(obs_list) < self.tri_min_views:
                print(f"[INFO] track {tid} has only {len(obs_list)} observations, skipping")
                continue
            # build cam centers and directions (in base frame)
            cam_centers = []
            directions = []
            for obs in obs_list:
                u, v = obs['u'], obs['v']
                T = obs['T_base_cam']
                C_base = T[:3, 3]
                R_base_cam = T[:3, :3]
                d_cam = pixel_to_ray(u, v, self.K)
                d_base = R_base_cam.dot(d_cam)#转换到世界坐标系下
                d_base = d_base / np.linalg.norm(d_base)
                cam_centers.append(C_base)
                directions.append(d_base)
            # robust triangulation
            X, mean_err, used_idx = robust_triangulate(cam_centers, directions, min_views=self.tri_min_views,
                                                       max_iter=self.tri_max_iter, thresh=self.tri_thresh)
            if X is None:
                print(f"[WARN] triangulation failed for track {tid}")
                continue
            # save result
            if len(used_idx)/frame_idx < 0.5:
                print(f"track{tid}:the number of used_idx less than thredhold, skip")
                continue
            results_3d.append({
                'track_id': tid,
                'X_base': X,
                'mean_error': mean_err,
                'num_obs': len(obs_list),
                'used_obs': len(used_idx)
            })
            print(f"[RESULT] track {tid} -> X_base = {X}, mean_err = {mean_err}, obs={len(obs_list)}, used={len(used_idx)}")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Object Centers and Camera Trajectory')

        # 绘制相机位姿轨迹
        if len(self.poses_array) > 0:
            cam_traj = np.array([T[:3, 3] for T in self.poses_array])
            ax.plot(cam_traj[:, 0], cam_traj[:, 1], cam_traj[:, 2], 'k--', lw=1, label='Camera Trajectory')
            # 每隔几帧绘制一个小坐标系
            for i in range(0, len(self.poses_array), max(1, len(self.poses_array)//15)):
                draw_axes(ax, self.poses_array[i], length=0.03)

        # 绘制每个目标的三维位置
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, temp in enumerate(results_3d):
            err=temp['mean_error']
            tid=temp['track_id']
            X=temp['X_base']
            c = colors[i % 10]
            err = float(err) if not isinstance(err, (float, np.floating)) else err
            ax.scatter(X[0], X[1], X[2], color=c, s=50, label=f'ID {tid}, err={err:.3f}m')

        ax.legend()
        ax.view_init(elev=25, azim=-60)
        plt.tight_layout()
        plt.savefig(self.save_plot_path, dpi=300)
        print(f"[INFO] 3D visualization saved to {self.save_plot_path}")
        plt.close(fig)

        # save results to CSV
        with open(self.out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['track_id','X_x','X_y','X_z','mean_err','num_obs','used_obs'])
            for r in results_3d:
                writer.writerow([r['track_id'], r['X_base'][0], r['X_base'][1], r['X_base'][2],
                                 r['mean_error'], r['num_obs'], r['used_obs']])
        print(f"[INFO] triangulation finished, results saved to {self.out_csv}")
        cv2.destroyAllWindows()
        return results_3d

# ---------------------
# CLI
# ---------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video', default="/home/dsj/code/ultralytics/apply/v2/v2.mp4")
    p.add_argument('--model', default='/home/dsj/code/ultralytics/apply/ultralytics/runs/train/yolov11_glassware/weights/best.pt')
    p.add_argument('--mode', choices=['offline','ros'], default='offline')
    p.add_argument('--poses', help='poses file (npy/npz/csv) for offline mode', default="/home/dsj/code/ultralytics/apply/v2/p2.txt")
    p.add_argument('--base_frame', default='base')
    p.add_argument('--camera_frame', default='camera_color_frame')
    p.add_argument('--min_views', type=int, default=3)
    p.add_argument('--Kfx', type=float, default=604.0)
    p.add_argument('--Kfy', type=float, default=603.7)
    p.add_argument('--Kcx', type=float, default=334.7)
    p.add_argument('--Kcy', type=float, default=250.7)
    p.add_argument('--out', default='/home/dsj/code/ultralytics/apply/triangulated_results.csv')
    return p.parse_args()

def main():
    args = parse_args()
    K = np.array([[args.Kfx, 0, args.Kcx],
                  [0, args.Kfy, args.Kcy],
                  [0, 0, 1]], dtype=np.float64)
    mt = MultiViewTriangulator(model_path=args.model, K=K, video_path=args.video,
                               mode=args.mode, poses_file=args.poses,
                               base_frame=args.base_frame, camera_frame=args.camera_frame,
                               min_views=args.min_views, out_csv=args.out)
    results = mt.process()
    print("All done.")

if __name__ == '__main__':
    main()
