import torch

ckpt = torch.load(
    "/home/dsj/code/ultralytics/apply/ultralytics/runs/train/yolov11_glassware_labdpics_seg/weights/last.pt",
    weights_only=False
)

print("训练停在第几轮:", ckpt['epoch'])

