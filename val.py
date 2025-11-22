import os
import cv2
import json
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics import FastSAMPrompt,FastSAM
from pycocotools.mask import encode,decode
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import matplotlib.pyplot as plt

def convert_numpy_types(obj):
    """递归将 NumPy 类型转换为 Python 原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # 数组转为列表
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj
import json
import os
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.mask import encode


class YOLOtoCOCOConverter:
    """
    将YOLO格式的边界框标签（labels文件夹）和分割掩码标签（labels_seg文件夹）转换为COCO格式。
    两类标签文件名与图像文件名一一对应（如image.jpg对应labels/image.txt和labels_seg/image.txt）。
    """
    def __init__(self, yolo_dataset_path):
        """
        初始化转换器。
        :param yolo_dataset_path: YOLO数据集根目录，结构应包含：
                                 - images/{split}/：图像文件
                                 - labels/{split}/：边界框标签（仅含类别和边界框）
                                 - labels_seg/{split}/：分割掩码标签（仅含类别和掩码多边形）
        :param split: 数据集分割（如'train'或'val'）
        """
        self.root = yolo_dataset_path
        
        # 定义图像和标签路径（分离的边界框和掩码标签）
        self.images_dir = os.path.join(self.root, 'instances_test2017/images',)
        self.bbox_labels_dir = os.path.join(self.root, 'instances_test2017/labels')  # 边界框标签
        self.seg_labels_dir = os.path.join(self.root, 'instances_test2017/labels_seg')  # 分割掩码标签
        
        # 加载类别名称（从dataset.yaml读取）
        self.classes = self._load_classes()
        
        # 初始化COCO格式字典
        self.coco_format = self._init_coco_format()

    def _load_classes(self):
        """从dataset.yaml加载类别名称"""
        yaml_path = os.path.join(self.root, 'mydata.yaml')
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"未找到数据集配置文件：{yaml_path}")
        
        import yaml
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if 'names' not in data:
            raise ValueError(f"dataset.yaml中未定义'names'字段（类别名称）")
        return data['names']

    def _init_coco_format(self):
        """初始化COCO格式的基础结构"""
        return {
            'info': {'description': f'YOLO格式转换（边界框与分割掩码分离）'},
            'licenses': [],
            'categories': [
                {'id': i + 1, 'name': cls, 'supercategory': 'none'} 
                for i, cls in enumerate(self.classes)
            ],
            'images': [],
            'annotations': []
        }

    def _yolo_bbox_to_coco(self, yolo_bbox, img_width, img_height):
        """
        将YOLO边界框（归一化x_center, y_center, w, h）转换为COCO格式（x1, y1, w, h，像素坐标）
        """
        x_center, y_center, w, h = yolo_bbox
        x1 = max(0.0, (x_center - w / 2) * img_width)
        y1 = max(0.0, (y_center - h / 2) * img_height)
        width = min(img_width - x1, w * img_width)
        height = min(img_height - y1, h * img_height)
        return [float(x1), float(y1), float(width), float(height)]

    def _yolo_seg_to_rle(self, yolo_mask, img_width, img_height):
        """
        将YOLO分割掩码（归一化多边形坐标）转换为COCO RLE格式，并计算掩码面积
        """
        # 掩码需至少3个点（6个值）
        if len(yolo_mask) < 6:
            return None, 0.0
        
        # 转换为像素坐标
        mask_points = np.array(yolo_mask).reshape(-1, 2).astype(np.float32)
        mask_points[:, 0] *= img_width  # x坐标
        mask_points[:, 1] *= img_height  # y坐标
        mask_points = mask_points.astype(np.int32)  # 转为整数像素坐标
        
        # 生成二值掩码
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.fillPoly(mask, [mask_points], 1)  # 填充多边形区域
        
        # 计算掩码面积（像素总数）
        mask_area = float(np.sum(mask))
        if mask_area <= 0:
            return None, 0.0
        
        # 编码为RLE格式
        rle = encode(np.asarray(mask, order="F"))  # COCO要求Fortran顺序
        rle['counts'] = rle['counts'].decode('utf-8')  # 转为字符串，确保JSON兼容
        return rle, mask_area

    def convert(self):
        """执行转换，返回COCO格式文件路径和图像列表"""
        image_id = 0
        annotation_id = 0
        image_list = []  # 存储图像ID、路径等信息
        
        # 遍历所有图像
        for img_name in tqdm(os.listdir(self.images_dir), desc=f"转换集为COCO格式"):
            # 过滤非图像文件
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            
            # 图像基本信息
            img_basename = os.path.splitext(img_name)[0]  # 不含扩展名的文件名（用于匹配标签）
            img_path = os.path.join(self.images_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告：无法读取图像 {img_path}，已跳过")
                continue
            img_height, img_width = img.shape[:2]
            image_id += 1
            
            # 添加图像信息到COCO
            self.coco_format['images'].append({
                'id': image_id,
                'file_name': img_name,
                'width': img_width,
                'height': img_height
            })
            image_list.append({
                'id': image_id,
                'path': img_path,
                'width': img_width,
                'height': img_height
            })
            
            # 1. 读取边界框标签（labels文件夹）
            bbox_label_path = os.path.join(self.bbox_labels_dir, f"{img_basename}.txt")
            if not os.path.exists(bbox_label_path):
                print(f"警告：边界框标签 {bbox_label_path} 不存在，已跳过该图像的标注转换")
                continue
            
            # 2. 读取分割掩码标签（labels_seg文件夹）
            seg_label_path = os.path.join(self.seg_labels_dir, f"{img_basename}.txt")
            if not os.path.exists(seg_label_path):
                print(f"警告：分割掩码标签 {seg_label_path} 不存在，已跳过该图像的标注转换")
                continue
            
            # 3. 解析边界框标签（存储为：类别ID -> 边界框）
            bbox_dict = {}  # key: cls_id, value: 边界框（yolo格式）
            with open(bbox_label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:  # 边界框标签格式：cls_id xc yc w h
                        print(f"警告：边界框标签格式错误（{bbox_label_path}），行：{line}，已跳过")
                        continue
                    try:
                        cls_id = int(parts[0])
                        bbox = list(map(float, parts[1:5]))
                        bbox_dict[cls_id] = bbox  # 按类别ID存储，确保与分割标签匹配
                    except (ValueError, IndexError) as e:
                        print(f"警告：解析边界框失败（{bbox_label_path}），行：{line}，错误：{e}")
            
            # 4. 解析分割掩码标签，并与边界框整合
            with open(seg_label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:  # 分割标签格式：cls_id x1 y1 x2 y2 ...（至少1个点）
                        print(f"警告：分割掩码标签格式错误（{seg_label_path}），行：{line}，已跳过")
                        continue
                    try:
                        cls_id = int(parts[0])
                        yolo_mask = list(map(float, parts[1:]))  # 掩码多边形（归一化坐标）
                        
                        # 检查是否有对应的边界框
                        if cls_id not in bbox_dict:
                            print(f"警告：分割掩码类别 {cls_id} 无对应边界框（{seg_label_path}），行：{line}，已跳过")
                            continue
                        yolo_bbox = bbox_dict[cls_id]
                        
                        # 转换边界框为COCO格式
                        coco_bbox = self._yolo_bbox_to_coco(yolo_bbox, img_width, img_height)
                        bbox_area = coco_bbox[2] * coco_bbox[3]
                        
                        # 转换掩码为RLE格式
                        segmentation, mask_area = self._yolo_seg_to_rle(yolo_mask, img_width, img_height)
                        if segmentation is None:
                            print(f"警告：无效掩码（{seg_label_path}），行：{line}，已跳过")
                            continue
                        
                        # 组装标注信息（优先使用掩码面积）
                        annotation_id += 1
                        self.coco_format['annotations'].append({
                            'id': annotation_id,
                            'image_id': image_id,
                            'category_id': cls_id + 1,  # COCO类别ID从1开始
                            'bbox': coco_bbox,
                            'area': mask_area if mask_area > 0 else bbox_area,
                            'segmentation': segmentation,
                            'iscrowd': 0
                        })
                        
                    except (ValueError, IndexError) as e:
                        print(f"警告：解析分割掩码失败（{seg_label_path}），行：{line}，错误：{e}")
        
        # 保存COCO格式文件
        coco_json_path = os.path.join(self.root, f'coco.json')
        try:
            with open(coco_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.coco_format, f, ensure_ascii=False, indent=2)
            print(f"COCO格式标注已保存至：{coco_json_path}")
        except IOError as e:
            print(f"错误：保存COCO标注文件失败，{e}")
            return None, image_list
        
        return coco_json_path, image_list



class InstanceSegmentationEvaluator:
    def __init__(self, yolo_dataset_path, fastsam_checkpoint,  device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化评估器（针对YOLO格式数据集）
        :param yolo_dataset_path: YOLO格式数据集根目录
        :param fastsam_checkpoint: FastSAM权重路径
        :param split: 评估集分割（'val'）
        :param device: 运行设备
        """
        self.device = device
        
        # 1. 转换YOLO格式到COCO格式
        self.converter = YOLOtoCOCOConverter(yolo_dataset_path)
        self.coco_gt_path, self.test_images = self.converter.convert()
        self.coco_gt = COCO(self.coco_gt_path)
        
        # 2. 加载模型
        self.yolov11 = YOLO('/home/dsj/code/ultralytics/apply/ultralytics/runs/train/yolov11_seg_baseline_modify/weights/best.pt').to(device)  # YOLOv11分割模型
        self.fastsam = FastSAM(fastsam_checkpoint).to(device)  # FastSAM模型
        self.fastsam_prompt = None

    def yolov11_direct_segment(self, image_path):
        """YOLOv11直接分割（输出掩码、边界框及面积，确保COCO兼容性）"""
        try:
            # 推理（确保使用分割模型）
            results = self.yolov11(image_path, device=self.device, conf=0.3)[0]
            predictions = []
            
            # 检查是否有掩码和边界框
            if results.masks is None or len(results.masks) == 0 or len(results.boxes) == 0:
                return predictions
            
            # 遍历结果（确保掩码和边界框数量一致）
            for mask, box, cls, score in zip(results.masks.data, results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                # 1. 处理掩码（转RLE格式）
                mask_np = mask.cpu().numpy().astype(np.uint8)  # 转为CPU numpy数组
                mask_rle = encode(np.asarray(mask_np, order="F"))  # 编码为RLE
                mask_rle['counts'] = mask_rle['counts'].decode('utf-8')  # 确保字符串类型
                
                # 2. 计算掩码面积（用于COCO评估的area字段）
                mask_area = np.sum(mask_np)  # 掩码像素总数
                
                # 3. 处理边界框（xyxy转xywh）
                box_np = box.cpu().numpy()  # 转为CPU numpy数组
                x1, y1, x2, y2 = box_np
                bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]  # 转为Python原生float
                
                # 4. 组装预测结果（补充所有必要字段）
                predictions.append({
                    'image_id': None,  # 后续由调用方填充
                    'category_id': int(cls.item()) + 1,  # 类别ID+1以对齐COCO格式
                    'bbox': bbox_xywh,
                    'score': float(score.item()),  # 置信度转为Python float
                    'segmentation': mask_rle,
                    'area': float(mask_area)  # 新增：掩码面积（COCO评估必需）
                })
            self.visualization(image_path,predictions,label="yolo")
            return predictions
        
        except Exception as e:
            print(f"YOLOv11直接分割出错（{image_path}）：{str(e)}")
            return []


    def yolov11_fastsam_segment(self, image_path):
        """YOLOv11检测边界框 + FastSAM分割（优化兼容性和错误处理）"""
        try:
            # 1. YOLOv11检测边界框（使用纯检测模型）
            det_results = self.yolov11(image_path, device=self.device, conf=0.3)[0]
            if len(det_results.boxes) == 0:
                return []  # 无检测结果则返回空
            
            # 2. 加载图像并验证
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法加载图像：{image_path}")
                return []
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]  # 获取图像尺寸
            '''
            # 3. FastSAM推理（无梯度计算，节省内存）
            with torch.no_grad():
                fastsam_results = self.fastsam(image_rgb, device=self.device, retina_masks=True, conf=0.4)
            
            # 初始化FastSAM提示工具
            self.fastsam_prompt = FastSAMPrompt(image_rgb, fastsam_results, device=self.device)
            '''
            predictions = []
            boxes = det_results.boxes.xyxy.cpu().numpy()  # xyxy格式（左上角x,y，右下角x,y）
            classes = det_results.boxes.cls.cpu().numpy()
            scores = det_results.boxes.conf.cpu().numpy()
            
            # 4. 为每个边界框生成掩码
            for box, cls, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = box
                # 确保边界框坐标在图像范围内（防止越界）
                x1 = int(max(0, min(x1, w)))
                y1 = int(max(0, min(y1, h)))
                x2 = int(max(0, min(x2, w)))
                y2 = int(max(0, min(y2, h)))
                
                # FastSAM边界框提示（生成对应掩码）
                #mask = self.fastsam_prompt.box_prompt(bboxes=[[x1, y1, x2, y2]])
                inference_params = {
                "device": "0",
                "retina_masks": True,
                "imgsz": 640,
                "conf": 0.4,
                "iou": 0.9,
                # 可选提示（按需取消注释）
                 "bboxes": [x1, y1, x2, y2],
                # "points": [[200, 200], [300, 300]],
                # "labels": [1, 0],  # 1:前景，0:背景
                #"texts": "the alcohol lamp"
    }
                fastsam_results = self.fastsam(image_path,** inference_params)
                mask=fastsam_results[0].masks.data
                if len(mask) == 0:
                    continue  # 无掩码则跳过

                # 5. 处理掩码（转RLE格式）
                mask_np = mask[0].cpu().numpy().astype(np.uint8)  # 取第一个掩码并转为uint8
                mask_rle = encode(np.asarray(mask_np, order="F"))
                mask_rle['counts'] = mask_rle['counts'].decode('utf-8')
                
                # 6. 计算掩码面积（COCO评估必需）
                mask_area = np.sum(mask_np)
                
                # 7. 处理边界框（xyxy转xywh）
                bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                
                # 8. 组装预测结果
                predictions.append({
                    'image_id': None,  # 后续填充
                    'category_id': int(cls) + 1,  # 类别ID对齐COCO
                    'bbox': bbox_xywh,
                    'score': float(score),  # 置信度转为Python float
                    'segmentation': mask_rle,
                    'area': float(mask_area)  # 新增：掩码面积
                })
            self.visualization(image_path,predictions,label="sam")
            return predictions
        
        except Exception as e:
            print(f"YOLOv11+FastSAM分割出错（{image_path}）：{str(e)}")
            return []

    def visualization(self, image_path, pre,label,save_dir="performance_fastsam_vs_yolo/visualization_results_labche"):
        """
        可视化两种分割方法的结果并保存对比图
        :param image_path: 测试图像路径
        :param save_dir: 结果保存目录（自动创建子目录区分两种方法）
        """
        # 创建保存目录
        if label=="sam":
            dir_fastsam = os.path.join(save_dir, "yolov11_fastsam")
            os.makedirs(dir_fastsam, exist_ok=True)
        if label=="yolo":
            dir_fastsam = os.path.join(save_dir, "direct_yolov11")
            os.makedirs(dir_fastsam, exist_ok=True)

        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告：无法读取图像 {image_path}，跳过可视化")
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_basename = os.path.splitext(os.path.basename(image_path))[0]  # 图像文件名（不含扩展名）

        '''
        # --------------------------
        # 1. 可视化YOLOv11直接分割结果
        # --------------------------
        direct_preds = self.yolov11(image_path)
        vis_direct = image_rgb.copy()  # 复制原图用于绘制

        for pred in direct_preds:
            # 解析预测结果
            bbox = pred['bbox']  # [x1, y1, w, h]
            mask_rle = pred['segmentation']
            cls_id = pred['category_id'] - 1  # 转回0基类别
            score = pred['score']

            # 绘制边界框
            x1, y1, w, h = map(int, bbox)
            cv2.rectangle(vis_direct, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)  # 绿色框

            # 绘制掩码（随机颜色）
            mask = decode(mask_rle)  # RLE解码为掩码
            if mask.ndim == 3:
                mask = mask.squeeze(2)  # 去除单通道维度
            color = np.random.randint(0, 255, 3).tolist()  # 随机颜色
            vis_direct[mask == 1] = vis_direct[mask == 1] * 0.5 + np.array(color) * 0.5  # 半透明叠加

            # 绘制类别和置信度
            cls_name = self.classes[cls_id] if hasattr(self, 'classes') else f"class_{cls_id}"
            cv2.putText(
                vis_direct,
                f"{cls_name}: {score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # 保存YOLOv11直接分割结果
        save_path_direct = os.path.join(dir_direct, f"{img_basename}.png")
        vis_direct_bgr = cv2.cvtColor(vis_direct.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path_direct, vis_direct_bgr)

        '''
        # --------------------------
        # 2. 可视化YOLOv11+FastSAM分割结果
        # --------------------------
        vis = image_rgb.copy()  # 复制原图用于绘制

        for pred in pre:
            # 解析预测结果
            bbox = pred['bbox']  # [x1, y1, w, h]
            mask_rle = pred['segmentation']
            cls_id = pred['category_id'] - 1  # 转回0基类别
            score = pred['score']

            # 绘制边界框
            x1, y1, w, h = map(int, bbox)
            cv2.rectangle(vis, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)  # 蓝色框

            # 绘制掩码（随机颜色）
            mask = decode(mask_rle)  # RLE解码为掩码
            if mask.ndim == 3:
                mask = mask.squeeze(2)  # 去除单通道维度
            color = np.random.randint(0, 255, 3).tolist()  # 随机颜色
            vis[mask == 1] = vis[mask == 1] * 0.5 + np.array(color) * 0.5  # 半透明叠加

            # 绘制类别和置信度
            cls_name = self.classes[cls_id] if hasattr(self, 'classes') else f"class_{cls_id}"
            cv2.putText(
                vis,
                f"{cls_name}: {score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

        save_path = os.path.join(dir_fastsam, f"{img_basename}.png")
        vis_bgr = cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, vis_bgr)

        print(f"可视化结果已保存：\n  - 分割: {save_path}")

    def evaluate(self, predictions):
        """用COCO指标评估"""

        predictions = convert_numpy_types(predictions)
        pred_json = 'yolo_predictions.json'
        with open(pred_json, 'w') as f:
            json.dump(predictions, f)
        
        coco_dt = self.coco_gt.loadRes(pred_json)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        return {
            'mAP@0.5': coco_eval.stats[1],
            'mAP@0.5:0.95': coco_eval.stats[0],
            'mAP_small': coco_eval.stats[2],
            'mAP_medium': coco_eval.stats[3],
            'mAP_large': coco_eval.stats[4],
            'AR@100': coco_eval.stats[6]
        }

    def run_comparison(self):
        """运行对比实验"""
        print("开始推理与评估...")
        yolov11_preds = []
        yolov11_fastsam_preds = []
        
        for img in tqdm(self.test_images, desc="Processing images"):
            img_id = img['id']
            img_path = img['path']
            
            # YOLOv11直接分割
            yolo_res = self.yolov11_direct_segment(img_path)
            for pred in yolo_res:
                pred['image_id'] = img_id
            yolov11_preds.extend(yolo_res)
            
            # YOLOv11+FastSAM分割
            fastsam_res = self.yolov11_fastsam_segment(img_path)
            for pred in fastsam_res:
                pred['image_id'] = img_id
            yolov11_fastsam_preds.extend(fastsam_res)
        
        # 评估
        print("\n评估YOLOv11直接分割：")
        yolo_metrics = self.evaluate(yolov11_preds)
        
        print("\n评估YOLOv11+FastSAM：")
        fastsam_metrics = self.evaluate(yolov11_fastsam_preds)
        
        # 可视化
        self.visualize(yolo_metrics, fastsam_metrics)
        return yolo_metrics, fastsam_metrics

    def visualize(self, yolo_metrics, fastsam_metrics):
        """可视化对比结果"""
        metrics = list(yolo_metrics.keys())
        yolo_vals = [v * 100 for v in yolo_metrics.values()]
        fastsam_vals = [v * 100 for v in fastsam_metrics.values()]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, yolo_vals, width, label='YOLOv11 Direct')
        plt.bar(x + width/2, fastsam_vals, width, label='YOLOv11 + FastSAM')
        
        plt.xlabel('Metrics')
        plt.ylabel('Score (%)')
        plt.title('YOLO Format Dataset - Segmentation Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.tight_layout()
        plt.savefig('yolo_dataset_comparison.png')
        print("\n对比图已保存为 yolo_dataset_comparison.png")
        #plt.show()


if __name__ == "__main__":
    # 配置参数（替换为你的实际路径）
    YOLO_DATASET_PATH = "/home/dsj/dataset/my_yolo"  # YOLO格式数据集根目录
    FASTSAM_CHECKPOINT = "/home/dsj/code/ultralytics/apply/FastSAM-x.pt"      # FastSAM权重路径
    
    # 初始化评估器并运行
    evaluator = InstanceSegmentationEvaluator(
        yolo_dataset_path=YOLO_DATASET_PATH,
        fastsam_checkpoint=FASTSAM_CHECKPOINT,
    )
    evaluator.run_comparison()
