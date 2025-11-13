import torch
from ultralytics import YOLO
from ultralytics.models.sam import SAM2DynamicInteractivePredictor
import cv2
import os

# ======================
# Step 1. 初始化模型
# ======================

# YOLO 检测模型（换成你训练好的模型）
yolo_model = YOLO("/home/dsj/code/ultralytics/apply/ultralytics/runs/train/yolov11_seg_baseline/weights/best.pt")

# SAM2 分割模型
overrides = dict(
    conf=0.01, 
    task="segment", 
    mode="predict", 
    imgsz=1024, 
    model="sam2_t.pt",  # 你可以换成 sam2_b.pt / sam2_h.pt
    save=False
)
sam2_predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)

# ======================
# Step 2. 定义输入输出路径
# ======================
image_path = "/home/dsj/dataset/my_yolo/instances_test2017/images/ce_000002.jpg"
output_dir = "./sam2_yolo_results"
os.makedirs(output_dir, exist_ok=True)

# ======================
# Step 3. 用 YOLO 检测
# ======================
results = yolo_model(image_path)
boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
classes = results[0].boxes.cls.cpu().numpy()  # 类别索引
confidences = results[0].boxes.conf.cpu().numpy()  # 置信度

print(f"检测到 {len(boxes)} 个目标")

# ======================
# Step 4. SAM2 分割（使用 YOLO 框作为提示）
# ======================
img = cv2.imread(image_path)

for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
    x1, y1, x2, y2 = map(int, box)

    # 用 YOLO 框提示 SAM2
    results_sam = sam2_predictor(
        source=img,
        bboxes=[[x1, y1, x2, y2]],  # 输入检测框作为提示
        obj_ids=[i],
        update_memory=True  # 让 SAM2 记住这个对象
    )

    # ======================
    # Step 5. 可视化与保存结果
    # ======================
    mask = results_sam[0].masks.data[0].cpu().numpy()  # 提取分割掩码
    mask = (mask * 255).astype("uint8")
    color = (0, 255, 0)

    overlay = img.copy()
    overlay[mask > 0] = (0, 255, 0)
    vis = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    save_path = os.path.join(output_dir, f"sam2_seg_{i}_cls{int(cls)}.jpg")
    cv2.imwrite(save_path, vis)
    print(f"已保存: {save_path}")

print("✅ YOLO + SAM2 分割完成！")
