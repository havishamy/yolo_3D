import cv2
from ultralytics import YOLO
import numpy as np

# 1️⃣ 加载模型
model = YOLO("/home/dsj/code/ultralytics/apply/ultralytics/runs/train/yolov11_glassware_labdpics_seg6/weights/best.pt")   # 可换 yolov8s.pt 或自己的模型

# 2️⃣ 打开视频源
# cap = cv2.VideoCapture(0)  # 摄像头
cap = cv2.VideoCapture("/home/dsj/code/ultralytics/apply/v2/v2.mp4")  # 视频文件

# 3️⃣ 初始化跟踪器（ByteTrack 内置）
model.track(
    source="/home/dsj/code/ultralytics/apply/v2/v2.mp4",  # 或 0 表示摄像头
    show=True,                # 实时显示
    tracker="bytetrack.yaml", # 使用 ByteTrack
    save=True,                # 是否保存输出视频
)
