import torch
from ultralytics import YOLO
from ultralytics.models.sam import SAM2DynamicInteractivePredictor
import cv2
import os
import numpy as np

# ======================
# Step 1. 初始化模型（仅用公开接口，不依赖私有变量）
# ======================
yolo_model = YOLO("/home/dsj/code/ultralytics/apply/ultralytics/runs/train/yolov11_seg_baseline_modify/weights/best.pt")

overrides = dict(
    conf=0.01,
    task="segment",
    mode="predict",
    imgsz=640,  # 适配 640×480 图像
    model="sam2_t.pt",
    save=False,
    device="0" if torch.cuda.is_available() else "cpu")
# 仅传入公开参数，不触碰内部属性
sam2_predictor = SAM2DynamicInteractivePredictor(overrides=overrides,max_obj_num=10)

# ======================
# Step 2. 输入输出配置
# ======================
image_path = "/home/dsj/dataset/my_yolo/instances_train2017/images/ce_000317.jpg"
output_dir = "./sam2_yolo_multi_id_results"
os.makedirs(output_dir, exist_ok=True)
img = cv2.imread(image_path)
h, w = img.shape[:2]  # 640×480

# ======================
# Step 3. YOLO 检测所有目标
# ======================
results = yolo_model(image_path)
boxes = results[0].boxes.xyxy.cpu().numpy()
classes = results[0].boxes.cls.cpu().numpy()
confidences = results[0].boxes.conf.cpu().numpy()
total_objs = len(boxes)
print(f"检测到 {total_objs} 个目标，开始多 ID 分割...")

# ======================
# Step 4. 核心逻辑：手动管理 ID，避开内部私有变量
# ======================
all_obj_info = []
# 用自己的字典记录 ID 与目标信息的映射（不依赖模型内部变量）
id_to_info = {}
box_all=[]
obj_ids=[]
for i in range(total_objs):
    box = boxes[i]
    cls = classes[i]
    conf = confidences[i]
    x1, y1, x2, y2 = map(int, box)
    
    # 过滤无效框
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    if x1 >= x2 or y1 >= y2:
        print(f"跳过无效框 {i}: [{x1}, {y1}, {x2}, {y2}]")
        continue
    box_all.append([x1, y1, x2, y2])
    obj_ids.append(i+1)
  
    # 显式传入 ID，update_memory=True 会自动关联 ID 与目标
results_sam = sam2_predictor(
    source=img,
    bboxes=box_all,
    obj_ids=obj_ids,  # 公开接口要求的参数，稳定可用
    update_memory=True
)
print(results_sam)

s=sam2_predictor(source="/home/dsj/dataset/my_yolo/instances_train2017/images/ce_000325.jpg")

# 提取掩码（单目标输入，返回结果中掩码索引固定为 0）
'''
if results_sam[0].masks is None:
    print(f"目标 {i}（ID:{obj_id}）未生成掩码，跳过")
    continue
    '''
num=results_sam[0].masks.data.shape[0]
for i in range(num):
    mask = results_sam[0].masks.data[i].cpu().numpy()
    valid_mask = (mask==1).astype("uint8")
    mask_area = np.sum(valid_mask)
    print(f"目标 {i}（ID:{i+1}）掩码大小：{mask_area}，开始裁剪...")
    mask = (mask * 255).astype("uint8")
        # 3. ✅ 检查掩码是否有大于0的有效区域
    if not np.any(mask > 0):
        print(f"目标 {i}（ID:{i+1}）掩码为空（无有效区域），跳过")
        continue


    # 用自己的字典存储，不依赖模型内部变量
    id_to_info[i+1] = {
        "obj_id": i+1,
        "mask": mask,
        "cls": int(classes[i]),
        "conf": confidences[i],
        "box": box_all[i]
    }
    all_obj_info.append(id_to_info[i+1])

    print(f"目标 {i} - 分配 ID:{i+1}, 类别:{int(classes[i])}, 置信度:{confidences[i]:.2f}")

# ======================
# Step 5. 可视化多 ID 结果
# ======================
vis_img = img.copy()
# 为每个 ID 分配固定颜色（循环使用）
colors = [(0,255,0),(0,0,255),(255,0,0),(255,255,0),(255,0,255),
          (0,255,255),(128,0,0),(0,128,0),(0,0,128),(128,128,0)]

for info in all_obj_info:
    obj_id = info["obj_id"]
    mask = info["mask"]
    x1, y1, x2, y2 = info["box"]
    cls = info["cls"]
    conf = info["conf"]
    # 按 ID 取颜色（确保同一 ID 颜色固定）
    color = colors[obj_id % len(colors)]

    # 叠加半透明掩码
    overlay = vis_img.copy()
    overlay[mask > 0] = color 
    vis_img = cv2.addWeighted(overlay, 0.5, vis_img, 0.5, 0)

    # 绘制框 + ID + 类别信息
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{obj_id} cls:{cls} conf:{conf:.2f}"
    cv2.putText(vis_img, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# 保存最终结果
final_save_path = os.path.join(output_dir, "sam2_multi_id_final.jpg")
cv2.imwrite(final_save_path, vis_img)
print(f"\n✅ 所有目标处理完成！")
print(f"成功分割 {len(all_obj_info)} 个目标，结果保存至: {final_save_path}")
print(f"分配的 ID 列表: {[info['obj_id'] for info in all_obj_info]}")
