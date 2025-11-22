from ultralytics import FastSAM
import cv2
import numpy as np
import os

def save_fastsam_results(model, source, save_path="fastsam_result.jpg",** inference_kwargs):
    """
    保存 FastSAM 分割结果为图片（不显示窗口）
    Args:
        model: FastSAM 模型实例
        source: 图像路径或 numpy 数组
        save_path: 结果保存路径（默认：fastsam_result.jpg）
        inference_kwargs: 推理参数（如 bboxes, points, texts 等）
    """
    # 1. 执行推理
    results = model(source, **inference_kwargs)
    # 获取原始图像（转为 BGR 格式用于 OpenCV 处理）
    image = cv2.imread(source) if isinstance(source, str) else source.copy()
    if image is None:
        raise ValueError("无法读取图像，请检查路径或输入格式")
    h, w = image.shape[:2]

    # 2. 准备掩码颜色（随机生成，带透明度）
    num_masks = len(results[0].masks.data) if results[0].masks is not None else 0
    colors = [
        (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        for _ in range(num_masks)
    ]  # (B, G, R)

    # 3. 创建可视化画布
    vis_image = image.copy()
    overlay = image.copy()  # 用于叠加掩码

    # 4. 绘制分割掩码和轮廓
    if results[0].masks is not None:
        for i, mask in enumerate(results[0].masks.data.cpu().numpy()):
            # 掩码 resize 到原图尺寸
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            # 绘制掩码（半透明叠加）
            b, g, r = colors[i]
            overlay[mask > 0] = (b, g, r)
            # 绘制掩码轮廓（绿色）
            contours, _ = cv2.findContours(
                (mask > 0).astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)

        # 融合掩码与原始图像（半透明效果）
        cv2.addWeighted(overlay, 0.5, vis_image, 0.5, 0, vis_image)

    # 5. 绘制提示信息（bboxes/points/texts）
    # 边界框提示（蓝色）
    if "bboxes" in inference_kwargs and inference_kwargs["bboxes"] is not None:
        bboxes = inference_kwargs["bboxes"]
        bboxes = [bboxes] if not isinstance(bboxes, list) else bboxes
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                vis_image, "bbox", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
            )

    # 点提示（红色：前景，绿色：背景）
    if "points" in inference_kwargs and inference_kwargs["points"] is not None:
        points = inference_kwargs["points"]
        labels = inference_kwargs.get("labels", [1] * len(points))
        for (x, y), label in zip(points, labels):
            color = (0, 0, 255) if label == 1 else (0, 255, 0)
            cv2.circle(vis_image, (int(x), int(y)), 5, color, -1)
            cv2.putText(
                vis_image, f"p{label}", (int(x) + 10, int(y) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    # 文本提示（黄色）
    if "texts" in inference_kwargs and inference_kwargs["texts"] is not None:
        texts = inference_kwargs["texts"]
        texts = [texts] if not isinstance(texts, list) else texts
        for i, text in enumerate(texts):
            cv2.putText(
                vis_image, f"text: {text}", (10, 30 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )

    # 6. 保存结果（不显示窗口）
    cv2.imwrite(save_path, vis_image)
    print(f"分割结果已保存至：{save_path}")

def batch_process(model,input_dir, output_dir, **inference_params):
    image_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"📂 批量处理 {len(image_list)} 张图片")
    for img_path in image_list:
        name=os.path.basename(img_path).split(".")[0]
        output_path=os.path.join(output_dir, f"{name}_mask.png")
        save_fastsam_results(model, img_path, output_path, **inference_params)

# 示例用法
if __name__ == "__main__":
    # 加载模型
    model = FastSAM("FastSAM-s.pt")  # 或 "FastSAM-x.pt"
    # 图像路径
    input_dir = "/home/dsj/dataset/my_yolo/instances_test2017/images"  # 替换为你的图像路径
    # 保存路径（可自定义，如 "output/segment_result.png"）
    output_dir = "/home/dsj/code/ultralytics/apply/fastsam_results"

    # 推理参数（根据需要调整）
    inference_params = {
        "device": "0",
        "retina_masks": True,
        "imgsz": 640,
        "conf": 0.4,
        "iou": 0.9,
        # 可选提示（按需取消注释）
        # "bboxes": [439, 437, 524, 709],
        # "points": [[200, 200], [300, 300]],
        # "labels": [1, 0],  # 1:前景，0:背景
        "texts": "the alcohol lamp"
    }

    # 保存结果（不显示窗口）
    batch_process(model, input_dir, output_dir,** inference_params)