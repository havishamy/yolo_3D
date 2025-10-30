import socket
import cv2
import numpy as np
import json
from ultralytics import YOLO

class YOLOTCParser:
    def __init__(self, listen_ip, listen_port):
        # TCP监听配置
        self.listen_ip = listen_ip  # 监听IP（0.0.0.0表示允许所有客户端连接）
        self.listen_port = listen_port  # 监听端口（需与客户端一致）
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.listen_ip, self.listen_port))
        self.server_socket.listen(1)
        print(f"服务器启动，监听 {self.listen_ip}:{self.listen_port}...")

        # YOLO模型加载（默认使用yolov8n，可替换为yolov8s/yolov8m等）
        self.model = YOLO("/home/dsj/code/ultralytics/apply/ultralytics/runs/train/yolov11_glassware/weights/best.pt")  # 自动下载预训练模型
        print("YOLO模型加载成功")

    def receive_image(self, client_socket):
        """接收客户端发送的图像"""
        # 先接收图像长度
        img_len_bytes = client_socket.recv(4)
        if not img_len_bytes:
            return None
        img_len = int.from_bytes(img_len_bytes, byteorder='big')

        # 接收图像数据（处理大文件分片接收，避免数据丢失）
        img_bytes = b""
        while len(img_bytes) < img_len:
            chunk = client_socket.recv(min(4096, img_len - len(img_bytes)))
            if not chunk:
                return None
            img_bytes += chunk

        # 解码图像
        encode_img = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(encode_img, cv2.IMREAD_COLOR)
        return image

    def send_result(self, client_socket, result):
        """将识别结果发送给客户端"""
        # JSON序列化结果
        result_json = json.dumps(result, ensure_ascii=False)
        result_bytes = result_json.encode('utf-8')
        result_len = len(result_bytes)

        # 先发送结果长度，再发送结果数据
        client_socket.sendall(result_len.to_bytes(4, byteorder='big'))
        client_socket.sendall(result_bytes)

    def run(self):
        """主循环：等待连接→接收图像→识别→回传结果"""
        while True:
            # 等待客户端连接
            client_socket, client_addr = self.server_socket.accept()
            print(f"客户端连接：{client_addr}")

            try:
                while True:
                    # 1. 接收图像
                    image = self.receive_image(client_socket)
                    if image is None:
                        print(f"客户端 {client_addr} 断开连接")
                        break

                    # 2. YOLO识别（conf=0.5过滤低置信度目标）
                    results = self.model(image, conf=0.5)

                    # 3. 解析识别结果
                    detections = []
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            # 提取坐标（x1,y1,x2,y2）、置信度、类别
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            confidence = float(box.conf[0].cpu().numpy())
                            class_name = self.model.names[int(box.cls[0].cpu().numpy())]
                            detections.append({
                                "class": class_name,
                                "confidence": confidence,
                                "bbox": [x1, y1, x2, y2]
                            })

                    # 4. 回传结果
                    result = {
                        "success": True,
                        "detections": detections,
                        "error": ""
                    }
                    self.send_result(client_socket, result)

            except Exception as e:
                print(f"处理客户端 {client_addr} 时出错：{str(e)}")
                # 发送错误信息给客户端
                error_result = {
                    "success": False,
                    "detections": [],
                    "error": str(e)
                }
                self.send_result(client_socket, error_result)
            finally:
                client_socket.close()

if __name__ == "__main__":
    # 监听所有IP，端口8888（需与客户端一致）
    SERVER = YOLOTCParser(listen_ip="0.0.0.0", listen_port=8888)
    SERVER.run()
