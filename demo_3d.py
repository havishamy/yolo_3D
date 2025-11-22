'''
我需要一个代码框架，下面我来进行详细描述，我的相机放在机械臂手上，刚开始会找到一个可以观察到曹最台面的全局视角，然后机械臂发生一段移动，这个过程保证相机一只能看到操作台面，但相机位姿发生改变。整个过程把realsnese相机拍摄到的图片传入我训练好的分割玻璃仪器的yolov11_seg模型中，实时分割玻璃仪器，然后在机械臂移动，相机视角发生变化的同时不停跟踪（track）检测分割到的多个玻璃仪器目标，然后将跟踪到的分割结果计算其分割掩码的质心，根据相机的位姿（相机在机械臂上可以事先标定）计算转换相机中心到物体质心在三维空间的射线，由于相机位姿一直在变化，可以实时累积并计算这些射线的交点，最终定位到物体质心在三维空间的确定位置和坐标。这个过程是实时的，比如在相机移动过程中捕获了三帧图片，就可以先计算粗略的三维质心位置，再继续移动获得更多视角的帧，得到更多的射线，筛掉偏离太大的射线，剩下的射线继续计算其交点（最小二乘法计算距所有射线距离最短的点）在相机移动视角变化的过程中不断增加新的帧（这个过程最好维护一个动态的数组，不断添加新的射线，丢弃太早的或者偏离太大的射线，然后不停的计算这个的动态数组里的射线的交点，达到不断更新操作台面场景中障碍物的三维质心位置）然后根据是别的结果对障碍物进行建模（知道三维中心位置，可以用圆柱体大致进行建模）方便后续机械臂在操作过过程中避开这些玻璃障碍物。请用python帮我写我上述描述好的代码框架。
'''
import cv2
import os
import numpy as np
import torch
import time
from collections import defaultdict, deque
from ultralytics import YOLO
from ultralytics.engine.results import Results
#import pyrealsense2 as rs  # RealSense SDK
from scipy.optimize import minimize  # 最小二乘法
from scipy.spatial import distance  # 距离计算
from scipy.spatial.transform import Rotation as R
from scipy import ndimage

# --------------------------
# 1. 配置参数（根据实际场景修改）
# --------------------------
class Config:
    # 模型配置
    MODEL_PATH = "/home/dsj/code/ultralytics/apply/ultralytics/runs/train/yolov11_seg_baseline_modify/weights/best.pt"  # 训练好的分割模型路径
    CONF_THRESH = 0.5  # 置信度阈值
    IOU_THRESH = 0.4  # NMS IoU阈值
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 推理设备

    # 跟踪配置
    TRACK_MAX_AGE = 5  # 目标最大消失帧数（超过则删除）
    TRACK_IOU_THRESH = 0.3  # 跟踪匹配IoU阈值
    MAX_TRACKED_TARGETS = 10  # 最大跟踪目标数

    # 3D定位配置
    RAY_QUEUE_MAX_LEN = 30  # 每个目标的最大射线缓存数（动态滑动窗口）
    MIN_RAYS_FOR_FILTER = 4 # 射线异常值筛选阈值（单位：米）
    RAY_DISTANCE_THRESH = 0.15
    MIN_RAYS_FOR_3D = 3  # 计算3D位置所需的最小射线数
    CYLINDER_RADIUS = 0.05  # 障碍物建模（圆柱体半径，单位：米）

    # RealSense相机配置
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30

    # 视频/位姿文件配置
    VIDEO_PATH = "/home/dsj/code/ultralytics/apply/v2/v2.mp4"  # RealSense录制的视频路径
    POSE_TXT_PATH = "/home/dsj/code/ultralytics/apply/v2/p2.txt"      # 每帧相机位姿的TXT文件路径
    SAVE_OUTPUT_TYPE = "video"              # 输出类型："video"（视频）或 "images"（图片序列）
    OUTPUT_DIR = "output_results"           # 结果保存目录（图片/视频均保存在此）
    OUTPUT_VIDEO_FPS = 30       

    # 相机内参（需通过标定获取，示例值）
    CAMERA_INTRINSIC = np.array([[604.0, 0, 334.7],
              [0, 603.7, 250.7],
              [0,   0,   1]])

# --------------------------
# 2. 核心类定义
# --------------------------
class GlassInstrumentTracker:
    """玻璃仪器分割、跟踪、3D定位核心类"""
    def __init__(self, config: Config):
        self.config = config
        self.model = self._load_model()  # 加载分割模型
        self.tracked_targets = defaultdict(dict)  # 跟踪目标缓存：{track_id: {rays, 3d_pos, mask, ...}}
        self.next_track_id = 0  # 下一个跟踪ID

    def _load_model(self):
        """加载YOLOv11分割模型"""
        model = YOLO(self.config.MODEL_PATH)
        model.to(self.config.DEVICE)
        print(f"模型加载成功，运行设备：{self.config.DEVICE}")
        return model

    def segment_frame(self, frame: np.ndarray) -> Results:
        """实时分割单帧图像，返回分割结果"""
        results = self.model(
            frame,
            conf=self.config.CONF_THRESH,
            iou=self.config.IOU_THRESH,
            device=self.config.DEVICE,
            verbose=False
        )[0]  # 取第一张图的结果
        return results

    def _compute_mask_centroid(self, mask: np.ndarray) -> tuple[float, float]:
        """计算分割掩码的2D质心（图像坐标系）"""
        # 找到掩码的所有非零像素
        '''
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) == 0 or len(y_coords) == 0:
            return (-1.0, -1.0)  # 无效掩码返回无效坐标
        # 计算质心
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        return (centroid_x, centroid_y)
        '''
        return  ndimage.center_of_mass(mask)

    def update_tracks(self, results: Results, camera_pose: np.ndarray):
        """更新目标跟踪，关联分割结果与历史跟踪目标，计算射线"""
        current_masks = results.masks.data.cpu().numpy() if results.masks is not None else []
        current_cls = results.boxes.cls.cpu().numpy() if results.boxes is not None else []
        current_boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []

        # 步骤1：计算当前帧所有检测目标的质心和掩码
        current_detections = []
        for mask, cls, box in zip(current_masks, current_cls, current_boxes):
            centroid_2d = self._compute_mask_centroid(mask)
            if centroid_2d[0] < 0:
                continue
            current_detections.append({
                "mask": mask,
                "cls": int(cls),
                "box": box,
                "centroid_2d": centroid_2d,
                "ray": self._compute_ray(centroid_2d, camera_pose)  # 计算3D射线
            })

        # 步骤2：关联当前检测与历史跟踪目标（IoU匹配）
        tracked_ids = list(self.tracked_targets.keys())
        matched_ids = set()

        for det in current_detections:
            best_iou = 0.0
            best_id = None
            # 与历史目标匹配
            for track_id in tracked_ids:
                if track_id in matched_ids:
                    continue
                hist_mask = self.tracked_targets[track_id]["last_mask"]
                iou = self._compute_mask_iou(det["mask"], hist_mask)
                if iou > self.config.TRACK_IOU_THRESH and iou > best_iou:
                    best_iou = iou
                    best_id = track_id

            if best_id is not None:
                # 匹配成功：更新历史目标
                self.tracked_targets[best_id]["last_mask"] = det["mask"]
                self.tracked_targets[best_id]["last_centroid_2d"] = det["centroid_2d"]
                self.tracked_targets[best_id]["last_box"] = det["box"]
                self.tracked_targets[best_id]["cls"] = det["cls"]
                self.tracked_targets[best_id]["age"] = 0  # 重置消失帧数
                # 添加新射线到缓存（滑动窗口）
                rays = self.tracked_targets[best_id]["rays"]
                rays.append(det["ray"])
                if len(rays) > self.config.RAY_QUEUE_MAX_LEN:
                    rays.popleft()  # 丢弃最早的射线
                # 筛选异常射线
                self.tracked_targets[best_id]["rays"] = self._filter_outlier_rays_by_distance(rays)
                # 计算3D位置（如果射线足够）
                if len(self.tracked_targets[best_id]["rays"]) >= self.config.MIN_RAYS_FOR_3D:
                    self.tracked_targets[best_id]["3d_pos"] = self._compute_3d_position(rays)
                matched_ids.add(best_id)
            else:
                # 匹配失败：新增跟踪目标
                if self.next_track_id < self.config.MAX_TRACKED_TARGETS:
                    self.tracked_targets[self.next_track_id] = {
                        "track_id": self.next_track_id,
                        "cls": det["cls"],
                        "last_mask": det["mask"],
                        "last_centroid_2d": det["centroid_2d"],
                        "last_box": det["box"],
                        "age": 0,
                        "rays": deque([det["ray"]], maxlen=self.config.RAY_QUEUE_MAX_LEN),
                        "3d_pos": None  # 初始无3D位置
                    }
                    self.next_track_id += 1

        # 步骤3：更新未匹配目标的消失帧数，超过阈值则删除
        for track_id in tracked_ids:
            if track_id not in matched_ids:
                self.tracked_targets[track_id]["age"] += 1
                if self.tracked_targets[track_id]["age"] > self.config.TRACK_MAX_AGE:
                    del self.tracked_targets[track_id]
                    print(f"删除跟踪目标：ID={track_id}（消失帧数超限）")

    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """计算两个掩码的IoU（用于跟踪匹配）"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0

    def _compute_ray(self, centroid_2d: tuple[float, float], camera_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        计算3D射线（相机中心到目标质心的射线）
        :param centroid_2d: 图像坐标系下的质心 (x, y)
        :param camera_pose: 相机位姿（4x4齐次矩阵，世界坐标系→相机坐标系）
        :return: 射线起点（世界坐标系）、射线方向向量（单位向量）
        """
        x, y = centroid_2d
        fx, fy = self.config.CAMERA_INTRINSIC[0, 0], self.config.CAMERA_INTRINSIC[1, 1]
        cx, cy = self.config.CAMERA_INTRINSIC[0, 2], self.config.CAMERA_INTRINSIC[1, 2]

        # 步骤1：图像坐标→相机坐标系（归一化）
        x_cam = (x - cx) / fx
        y_cam = (y - cy) / fy
        z_cam = 1.0  # 归一化深度（射线方向）
        ray_dir_cam = np.array([x_cam, y_cam, z_cam])
        ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)  # 单位向量

        # 步骤2：相机坐标系→世界坐标系
        camera_pos_world = camera_pose[:3, 3]  # 相机在世界坐标系的位置（射线起点）
        rotation_matrix = camera_pose[:3, :3]  # 相机旋转矩阵
        ray_dir_world = rotation_matrix @ ray_dir_cam  # 射线方向转换到世界坐标系
        ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)  # 单位向量

        return (camera_pos_world, ray_dir_world)

    def _filter_outlier_rays_by_distance(self, rays: deque) -> deque:
        """
        基于「射线到汇聚点的距离」筛选离群射线
        逻辑：先算临时汇聚点 → 计算每条射线到该点的垂直距离 → 剔除距离超阈值的射线
        """
        # 射线数量不足时，不筛选（避免临时交点不可靠）
        if len(rays) < self.config.MIN_RAYS_FOR_FILTER:
            return rays

        # 步骤1：计算临时汇聚点（所有射线的初始交点）
        temp_3d_pos = self._compute_3d_position(rays)
        if temp_3d_pos is None:
            return rays  # 交点计算失败，返回原始射线

        # 步骤2：计算每条射线到临时汇聚点的垂直距离
        ray_distances = []
        for (ray_origin, ray_dir) in rays:
            # 点到射线的垂直距离公式：||(P - O) × dir|| / ||dir||
            vec_op = temp_3d_pos - ray_origin
            cross = np.cross(vec_op, ray_dir)
            dist = np.linalg.norm(cross) / np.linalg.norm(ray_dir)
            ray_distances.append(dist)

        # 步骤3：筛选距离 ≤ 阈值的射线（保留汇聚性好的射线）
        valid_indices = [
            i for i, dist in enumerate(ray_distances)
            if dist <= self.config.RAY_DISTANCE_THRESH
        ]

        # 步骤4：确保筛选后射线数量不低于最小要求（避免过度筛选）
        if len(valid_indices) < self.config.MIN_RAYS_FOR_3D:
            # 若有效射线太少，取距离最小的前N条（N=MIN_RAYS_FOR_3D）
            sorted_indices = np.argsort(ray_distances)[:self.config.MIN_RAYS_FOR_3D]
            valid_rays = deque([rays[i] for i in sorted_indices], maxlen=self.config.RAY_QUEUE_MAX_LEN)
            print(f"射线筛选后数量不足，保留距离最小的{self.config.MIN_RAYS_FOR_3D}条射线")
        else:
            valid_rays = deque([rays[i] for i in valid_indices], maxlen=self.config.RAY_QUEUE_MAX_LEN)

        # 输出筛选信息
        outlier_count = len(rays) - len(valid_rays)
        if outlier_count > 0:
            print(f"筛选掉{outlier_count}条离群射线（距离阈值：{self.config.RAY_DISTANCE_THRESH}m）")

        return valid_rays


    def _compute_3d_position(self, rays: deque) -> np.ndarray:
        """
        最小二乘法计算多条射线的最佳交点（距所有射线距离最短的点）
        :param rays: 射线列表，每条射线格式：(起点, 方向向量)
        :return: 3D目标位置（世界坐标系）
        """
        def distance_to_rays(point, rays):
            """计算点到所有射线的距离之和"""
            total_dist = 0.0
            for (ray_origin, ray_dir) in rays:
                # 点到射线的距离公式：||(P - O) × dir|| / ||dir||
                vec_op = point - ray_origin
                cross = np.cross(vec_op, ray_dir)
                dist = np.linalg.norm(cross) / np.linalg.norm(ray_dir)
                total_dist += dist ** 2  # 平方和（便于优化）
            return total_dist

        # 初始猜测：所有射线起点的均值
        ray_origins = np.array([ray[0] for ray in rays])
        initial_guess = np.mean(ray_origins, axis=0)

        # 最小化距离之和
        result = minimize(
            fun=distance_to_rays,
            x0=initial_guess,
            args=(rays,),
            method="L-BFGS-B"
        )

        if result.success:
            return result.x  # 最优3D位置
        else:
            print(f"3D位置计算失败：{result.message}")
            return initial_guess  # 失败时返回初始猜测
    
    def draw_results_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """在帧上绘制分割、跟踪、3D定位结果（用于保存）"""
        vis_frame = frame.copy()
        for track_id, target in self.tracked_targets.items():
            mask = target["last_mask"]
            box = target["last_box"]
            centroid_2d = target["last_centroid_2d"]
            cls = target["cls"]
            pos_3d = target["3d_pos"]

            # 绘制掩码（半透明）
            color = np.random.randint(0, 255, 3).tolist()
            vis_frame[mask == 1] = vis_frame[mask == 1] * 0.5 + np.array(color) * 0.5

            # 绘制边界框和质心
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis_frame, (int(centroid_2d[1]), int(centroid_2d[0])), 5, (0, 255, 0), -1)

            # 绘制跟踪ID和3D位置
            if pos_3d is not None:
                label = f"ID:{track_id} | 3D: ({pos_3d[0]:.2f}, {pos_3d[1]:.2f}, {pos_3d[2]:.2f})"
            else:
                label = f"ID:{track_id} | 3D: 计算中"
            cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return vis_frame


class ObstacleModeler:
    """障碍物建模（圆柱体建模）"""
    def __init__(self, config: Config):
        self.config = config
        self.obstacles = {}  # 障碍物缓存：{track_id: {3d_pos, radius, cls, ...}}

    def update_obstacles(self, tracked_targets: dict):
        """根据跟踪结果更新障碍物模型"""
        current_obstacle_ids = set()
        for track_id, target in tracked_targets.items():
            if target["3d_pos"] is not None:
                # 圆柱体建模：中心=3D质心，半径=配置值，高度=默认值（可根据实际场景调整）
                self.obstacles[track_id] = {
                    "track_id": track_id,
                    "cls": target["cls"],
                    "3d_center": target["3d_pos"],
                    "radius": self.config.CYLINDER_RADIUS,
                    "height": 0.2,  # 圆柱体高度（单位：米，可根据仪器尺寸调整）
                    "last_update_time": time.time()
                }
                current_obstacle_ids.add(track_id)

        # 删除长时间未更新的障碍物
        expired_ids = []
        for track_id in self.obstacles.keys():
            if track_id not in current_obstacle_ids:
                expired_ids.append(track_id)
        for track_id in expired_ids:
            del self.obstacles[track_id]
            print(f"删除过期障碍物：ID={track_id}")
    
    def set_current_frame_idx(self, frame_idx: int):
        """设置当前帧号（用于过期判断）"""
        self.current_frame_idx = frame_idx

    def get_obstacles(self) -> list[dict]:
        """获取当前所有有效障碍物模型"""
        return list(self.obstacles.values())

'''
class RealSenseCamera:
    """RealSense相机接口（获取彩色图+相机位姿）"""
    def __init__(self, config: Config):
        self.config = config
        self.pipeline = rs.pipeline()
        self.config_rs = rs.config()
        self._init_camera()

    def _init_camera(self):
        """初始化RealSense相机"""
        # 配置彩色流
        self.config_rs.enable_stream(
            rs.stream.color,
            self.config.CAMERA_WIDTH,
            self.config.CAMERA_HEIGHT,
            rs.format.bgr8,
            self.config.CAMERA_FPS
        )
        # 启动相机
        profile = self.pipeline.start(self.config_rs)
        print("RealSense相机启动成功")

    def get_frame_and_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """
        获取单帧彩色图和相机位姿
        :return: (彩色图, 相机位姿4x4齐次矩阵)
        """
        # 等待帧数据
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("获取相机帧失败")

        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())

        # 获取相机位姿（需根据机械臂标定结果替换！这里用示例位姿）
        # 实际场景：从机械臂API获取末端执行器（相机）的位姿（世界坐标系→相机坐标系）
        camera_pose = self._get_simulated_camera_pose()  # 模拟位姿，需替换为真实位姿

        return color_image, camera_pose

    def _get_simulated_camera_pose(self) -> np.ndarray:
        """
        模拟相机位姿（实际场景需替换为机械臂API获取的真实位姿）
        这里模拟相机缓慢移动，生成不同视角的位姿
        """
        t = time.time() * 0.5  # 缓慢变化的时间因子
        # 相机位置（x,y,z）缓慢移动
        x = 0.5 + 0.1 * np.sin(t)
        y = 0.3 + 0.1 * np.cos(t)
        z = 0.8  # 高度固定
        # 相机旋转矩阵（简单绕z轴旋转）
        yaw = 0.1 * np.sin(t)  # 偏航角
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        # 4x4齐次矩阵（世界→相机）
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = rotation_matrix
        camera_pose[:3, 3] = np.array([x, y, z])
        return camera_pose

    def release(self):
        """释放相机资源"""
        self.pipeline.stop()
        print("相机已关闭")
'''
class VideoPoseReader:
    """读取视频文件和对应的相机位姿TXT文件"""
    def __init__(self, video_path: str, pose_txt_path: str):
        self.video_path = video_path
        self.pose_txt_path = pose_txt_path
        self.cap = None
        self.total_frames = 0
        self.current_frame_idx = 0

        # 初始化：读取视频和位姿
        self._load_video()
        self._load_poses_from_txt()

    def _load_video(self):
        """加载视频文件"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"无法打开视频文件：{self.video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"视频加载成功：总帧数={self.total_frames}，FPS={self.video_fps:.1f}")

    def _load_poses_from_txt(self):
        """从TXT文件加载相机位姿（每行对应一帧的4x4齐次矩阵）"""
        if not os.path.exists(self.pose_txt_path):
            raise FileNotFoundError(f"位姿TXT文件不存在：{self.pose_txt_path}")

        
        self.poses_array = np.loadtxt(self.pose_txt_path, delimiter=",",skiprows=1)
        poses_list=[]
        # 解析每一行的位姿（假设每行16个浮点数，对应4x4矩阵的按行展开）
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


        # 验证位姿数量与视频帧数一致
        if len(self.poses_array) != self.total_frames:
            raise ValueError(f"位姿数量与视频帧数不匹配：位姿{len(self.poses_array)}帧，视频{self.total_frames}帧")
        print(f"位姿加载成功：共{len(self.poses_array)}帧位姿")

    def read_frame_and_pose(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """读取下一帧图像和对应的相机位姿"""
        if self.current_frame_idx >= self.total_frames:
            return None, None, False  # 读取完毕

        # 读取视频帧
        ret, frame = self.cap.read()
        if not ret:
            return None, None, False

        # 获取对应帧的位姿
        pose = self.poses_array[self.current_frame_idx]
        self.current_frame_idx += 1
        return frame, pose, True

    def release(self):
        """释放资源"""
        if self.cap is not None:
            self.cap.release()
        print("视频资源已释放")


# --------------------------
# 5. 结果保存工具（新增）
# --------------------------
class ResultSaver:
    """保存结果为视频或图片序列"""
    def __init__(self, config: Config, frame_width: int, frame_height: int):
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.video_writer = None
        self.image_dir = None

        # 创建输出目录
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        # 初始化保存工具
        if config.SAVE_OUTPUT_TYPE == "video":
            self._init_video_writer()
        elif config.SAVE_OUTPUT_TYPE == "images":
            self._init_image_dir()
        else:
            raise ValueError(f"无效的输出类型：{config.SAVE_OUTPUT_TYPE}，仅支持'video'或'images'")

    def _init_video_writer(self):
        """初始化视频写入器"""
        video_path = os.path.join(self.config.OUTPUT_DIR, "output_result.mp4")
        # 视频编码格式（MP4推荐用mp4v）
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            self.config.OUTPUT_VIDEO_FPS,
            (self.frame_width, self.frame_height)
        )
        if not self.video_writer.isOpened():
            raise RuntimeError("视频写入器初始化失败")
        print(f"视频保存已初始化：{video_path}（FPS={self.config.OUTPUT_VIDEO_FPS}）")

    def _init_image_dir(self):
        """初始化图片保存目录"""
        self.image_dir = os.path.join(self.config.OUTPUT_DIR, "output_images")
        os.makedirs(self.image_dir, exist_ok=True)
        print(f"图片序列保存已初始化：{self.image_dir}（帧号从0开始）")

    def save_frame(self, frame: np.ndarray, frame_idx: int):
        """保存单帧结果"""
        if self.config.SAVE_OUTPUT_TYPE == "video":
            self.video_writer.write(frame)
        elif self.config.SAVE_OUTPUT_TYPE == "images":
            image_path = os.path.join(self.image_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(image_path, frame)

    def release(self):
        """释放资源"""
        if self.video_writer is not None:
            self.video_writer.release()
        print("结果保存资源已释放")
# --------------------------
# 3. 主程序（实时运行逻辑）
# --------------------------
'''              
def main():
    # 初始化组件
    config = Config()
    #camera = RealSenseCamera(config)
    tracker = GlassInstrumentTracker(config)
    modeler = ObstacleModeler(config)

    try:
        print("开始实时分割、跟踪与3D定位...（按'q'退出）")
        while True:
            start_time = time.time()

            # 步骤1：获取相机帧和位姿
            color_frame, camera_pose = camera.get_frame_and_pose()

            # 步骤2：实时分割
            seg_results = tracker.segment_frame(color_frame)

            # 步骤3：更新目标跟踪和射线计算
            tracker.update_tracks(seg_results, camera_pose)

            # 步骤4：更新障碍物建模
            modeler.update_obstacles(tracker.tracked_targets)

            # 步骤5：可视化结果（可选，便于调试）
            vis_frame = tracker.visualize_results(color_frame, seg_results, tracker.tracked_targets, modeler.obstacles)

            # 步骤6：输出实时信息
            fps = 1 / (time.time() - start_time)
            print(f"FPS: {fps:.1f} | 跟踪目标数: {len(tracker.tracked_targets)} | 障碍物数: {len(modeler.obstacles)}")

            # 显示可视化窗口
            cv2.imshow("Real-Time Segmentation & Tracking", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        # 释放资源
        camera.release()
        cv2.destroyAllWindows()
        print("程序正常退出")


# --------------------------
# 4. 可视化辅助方法（添加到GlassInstrumentTracker类）
# --------------------------
def visualize_results(self, frame: np.ndarray, seg_results: Results, tracked_targets: dict, obstacles: list[dict]) -> np.ndarray:
    """可视化分割、跟踪、3D定位结果"""
    vis_frame = frame.copy()

    # 绘制分割掩码和跟踪框
    for track_id, target in tracked_targets.items():
        mask = target["last_mask"]
        box = target["last_box"]
        centroid_2d = target["last_centroid_2d"]
        cls = target["cls"]
        threeD_pos = target["3d_pos"]

        # 绘制掩码（半透明叠加）
        color = np.random.randint(0, 255, 3).tolist()
        vis_frame[mask == 1] = vis_frame[mask == 1] * 0.5 + np.array(color) * 0.5

        # 绘制边界框
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

        # 绘制质心
        cv2.circle(vis_frame, (int(centroid_2d[0]), int(centroid_2d[1])), 5, (0, 255, 0), -1)

        # 绘制跟踪ID和3D位置
        if threeD_pos is not None:
            label = f"ID:{track_id} | 3D: ({threeD_pos[0]:.2f}, {threeD_pos[1]:.2f}, {threeD_pos[2]:.2f})"
        else:
            label = f"ID:{track_id} | 3D: 计算中"
        cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return vis_frame

# 动态添加可视化方法到类
GlassInstrumentTracker.visualize_results = visualize_results
'''
def main():
    config = Config()

    try:
        # 步骤1：初始化组件
        # 加载视频和位姿
        video_pose_reader = VideoPoseReader(config.VIDEO_PATH, config.POSE_TXT_PATH)
        # 获取视频帧尺寸（用于初始化结果保存器）
        frame_width = int(video_pose_reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_pose_reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 初始化跟踪器、建模器、结果保存器
        tracker = GlassInstrumentTracker(config)
        modeler = ObstacleModeler(config)
        result_saver = ResultSaver(config, frame_width, frame_height)

        # 步骤2：逐帧处理
        print("\n开始逐帧处理...（按Ctrl+C中断）")
        start_time = time.time()
        while True:
            # 读取当前帧和对应的相机位姿
            frame, camera_pose, ret = video_pose_reader.read_frame_and_pose()
            if not ret:
                break  # 所有帧处理完毕

            current_frame_idx = video_pose_reader.current_frame_idx - 1  # 当前帧号（从0开始）

            # 实时分割
            seg_results = tracker.segment_frame(frame)

            # 更新目标跟踪和3D定位
            tracker.update_tracks(seg_results, camera_pose)

            # 更新障碍物建模（传入当前帧号）
            modeler.set_current_frame_idx(current_frame_idx)
            modeler.update_obstacles(tracker.tracked_targets)

            # 在帧上绘制结果（用于保存）
            result_frame = tracker.draw_results_on_frame(frame)

            # 保存结果帧
            result_saver.save_frame(result_frame, current_frame_idx)

            # 输出进度信息（每10帧打印一次）
            if current_frame_idx % 10 == 0:
                elapsed_time = time.time() - start_time
                progress = (current_frame_idx + 1) / video_pose_reader.total_frames * 100
                fps = (current_frame_idx + 1) / elapsed_time
                print(f"进度：{current_frame_idx+1}/{video_pose_reader.total_frames}（{progress:.1f}%）| FPS：{fps:.1f} | 跟踪目标数：{len(tracker.tracked_targets)} | 障碍物数：{len(modeler.obstacles)}")

        # 步骤3：处理完毕
        total_elapsed_time = time.time() - start_time
        print(f"\n所有帧处理完毕！总耗时：{total_elapsed_time:.2f}s | 平均FPS：{video_pose_reader.total_frames / total_elapsed_time:.1f}")
        print(f"结果保存路径：{config.OUTPUT_DIR}")

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序出错：{str(e)}")
    finally:
        # 释放所有资源
        if 'video_pose_reader' in locals():
            video_pose_reader.release()
        if 'result_saver' in locals():
            result_saver.release()
        print("程序正常退出")

# --------------------------
# 程序入口
# --------------------------
if __name__ == "__main__":
    main()
