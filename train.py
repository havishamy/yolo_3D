from ultralytics import YOLO

def main():
    # 1. 加载模型（选择合适的规模：n/s/m/l/x）
    # 这里用 YOLOv11n（速度快，轻量），你也可以换成 'yolov11s.pt' 等
    model = YOLO("/home/dsj/code/ultralytics/yolo11n.pt")

    # 2. 开始训练
    results = model.train(
        data="/home/dsj/code/ultralytics/ultralytics/cfg/datasets/CLAD.yaml",   # 数据集配置文件路径
        epochs=500,                 # 训练轮数
        batch=16,                   # batch size，根据显存调整
        imgsz=640,                  # 输入图片大小
        workers=4,                  # 数据加载线程数
        device=[4,5,6,7],      # 训练设备，0 表示第一张 GPU
        project="/home/dsj/code/ultralytics/ultralytics/runs/train",       # 保存日志的目录
        name="yolov11_glassware",   # 实验名字
        pretrained=True,            # 使用预训练权重
        optimizer="SGD",            # 也可以选 Adam/AdamW
        lr0=0.01,                   # 初始学习率
        cos_lr=True,                # 是否使用余弦退火学习率调度
        patience=100,                # EarlyStopping 提前停止
        save_period=10              # 每隔多少 epoch 保存一次权重
    )

    # 3. 打印训练结果
    print("训练完成，结果保存在:", results.save_dir)


if __name__ == "__main__":
    main()
