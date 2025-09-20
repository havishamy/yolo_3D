import cv2
from ultralytics import YOLO

# 1. 加载训练好的 YOLOv11 模型
# 如果你训练好的模型在 runs/detect/train/weights/best.pt
model = YOLO("runs/detect/train/weights/best.pt")

# 2. 输入图片（可以换成摄像头或视频）
img_path = "test.jpg"  # 换成你的测试图片路径
img = cv2.imread(img_path)

# 3. 推理
results = model(img)

# 4. 遍历检测结果
for r in results:
    boxes = r.boxes  # 检测到的所有框
    for box in boxes:
        # 取出坐标 (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = model.names[cls]

        # 计算几何中心
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # 画框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # 画中心点
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

        # 标注类别 + 置信度
        cv2.putText(img, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# 5. 显示结果
cv2.imshow("Detection with Centers", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. 保存结果
cv2.imwrite("result.jpg", img)


'''
cap = cv2.VideoCapture(0)  # 摄像头
# cap = cv2.VideoCapture("test.mp4")  # 视频

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

'''
