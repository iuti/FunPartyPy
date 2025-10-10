import cv2
import torch
from ultralytics import YOLO
import numpy as np
import socket
import json
import time
import math
from pathlib import Path


class EllipseRegion:
    def __init__(self, config_path: Path | str, frame_width: int, frame_height: int):
        self.valid = False
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.cos_angle = 1.0
        self.sin_angle = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.a = 0.0
        self.b = 0.0
        self.polygon = None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"[EllipseRegion] calibration config not found at {config_path}, ellipse filter disabled.")
            return
        except json.JSONDecodeError as exc:
            print(f"[EllipseRegion] failed to parse calibration config: {exc}. Ellipse filter disabled.")
            return

        ellipse = data.get("ellipse")
        frame_info = data.get("frame") or {}
        if not ellipse:
            print("[EllipseRegion] ellipse data missing in calibration config. Ellipse filter disabled.")
            return

        calib_width = frame_info.get("width")
        calib_height = frame_info.get("height")
        if not calib_width or not calib_height:
            print("[EllipseRegion] frame dimensions missing in calibration config. Ellipse filter disabled.")
            return

        self.scale_x = frame_width / calib_width
        self.scale_y = frame_height / calib_height
        if self.scale_x <= 0 or self.scale_y <= 0:
            print("[EllipseRegion] invalid scale factors derived from calibration config. Ellipse filter disabled.")
            return

        self.cx = float(ellipse.get("center_x", 0.0))
        self.cy = float(ellipse.get("center_y", 0.0))
        self.a = float(ellipse.get("major_axis", 0.0)) / 2.0
        self.b = float(ellipse.get("minor_axis", 0.0)) / 2.0
        angle_deg = float(ellipse.get("angle_deg", 0.0))

        if self.a <= 0 or self.b <= 0:
            print("[EllipseRegion] invalid ellipse axes in calibration config. Ellipse filter disabled.")
            return

        angle_rad = math.radians(angle_deg)
        self.cos_angle = math.cos(angle_rad)
        self.sin_angle = math.sin(angle_rad)

        # Precompute polygon for drawing in current frame coordinates
        points = []
        for deg in range(0, 360, 2):
            phi = math.radians(deg)
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)
            x_calib = self.cx + self.a * cos_phi * self.cos_angle - self.b * sin_phi * self.sin_angle
            y_calib = self.cy + self.a * cos_phi * self.sin_angle + self.b * sin_phi * self.cos_angle
            x_cur = int(round(x_calib * self.scale_x))
            y_cur = int(round(y_calib * self.scale_y))
            points.append([x_cur, y_cur])

        if points:
            self.polygon = np.array(points, dtype=np.int32)

        self.valid = True
        print("[EllipseRegion] ellipse filter enabled.")

    def contains(self, x: float, y: float) -> bool:
        if not self.valid:
            return True

        x_calib = x / self.scale_x
        y_calib = y / self.scale_y
        dx = x_calib - self.cx
        dy = y_calib - self.cy

        x_rot = dx * self.cos_angle + dy * self.sin_angle
        y_rot = -dx * self.sin_angle + dy * self.cos_angle

        value = (x_rot / self.a) ** 2 + (y_rot / self.b) ** 2
        return value <= 1.0

    def draw(self, frame, color=(0, 255, 255), thickness=2):
        if not self.valid or self.polygon is None:
            return
        cv2.polylines(frame, [self.polygon], isClosed=True, color=color, thickness=thickness)

# GPU / デバイス設定
use_cuda = torch.cuda.is_available()
device_str = "cuda:0" if use_cuda else "cpu"
print(f"Torch CUDA available: {use_cuda}")

# モデル読み込み（利用可能ならGPUへ）
try:
    model = YOLO("yolov8n.pt")
    # move underlying PyTorch model to device if possible
    if use_cuda:
        try:
            if hasattr(model, 'to'):
                try:
                    model.to(device_str)
                except Exception:
                    pass
            if hasattr(model, 'model'):
                try:
                    model.model.to(device_str)
                except Exception:
                    pass
            print(f"Model moved to {device_str}")
        except Exception:
            pass
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    model = YOLO("yolov8n.pt")

# UDP通信設定
UDP_IP = "127.0.0.1"  # Unityが動作するIPアドレス（localhost）
UDP_PORT = 12345      # Unityで受信するポート番号
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(2)

# カメラの解像度を取得
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"カメラ解像度: {frame_width}x{frame_height}")

ellipse_region = EllipseRegion(Path(__file__).parent / "calibration_config.json", frame_width, frame_height)

# トラッキング用の色を準備（より多くの色を用意）
colors = np.random.randint(0, 255, size=(50, 3), dtype="uint8")

while True:
    ret, frame = cap.read()
    if not ret:
        print("カメラからフレームを取得できませんでした")
        break
    
    # トラッキング実行（persist=Trueで同一人物にIDを維持）
    results = model.track(frame, persist=True, classes=[0], conf=0.5)
    
    # Unity送信用のデータリスト
    persons_data = []
    
    if results[0].boxes is not None:
        # バウンディングボックス、信頼度、トラックIDを取得
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # トラックIDが利用可能かチェック
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        else:
            # トラックIDが利用できない場合は連番を使用
            track_ids = list(range(len(boxes)))
        
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = map(int, box)
            
            # 足元の座標を計算（バウンディングボックスの下辺中央）
            foot_x = int((x1 + x2) / 2)
            foot_y = y2  # バウンディングボックスの底辺

            if not ellipse_region.contains(foot_x, foot_y):
                continue
            
            # 座標を正規化（0-1の範囲に変換）
            normalized_x = foot_x / frame_width
            normalized_y = foot_y / frame_height
            
            # トラックIDに基づいて色を選択
            track_id = track_ids[i] if i < len(track_ids) else i
            color = [int(c) for c in colors[track_id % len(colors)]]
            
            # Unity送信用データに追加
            person_data = {
                "id": int(track_id),
                "x": float(normalized_x),
                "y": float(normalized_y),
                "foot_x_pixel": int(foot_x),
                "foot_y_pixel": int(foot_y),
                "confidence": float(conf)
            }
            persons_data.append(person_data)
            
            # バウンディングボックスを描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 足元の位置に円を描画
            cv2.circle(frame, (foot_x, foot_y), 5, color, -1)
            
            # トラックIDと座標を含むラベルを表示
            label = f"ID:{track_id} ({conf:.2f})"
            coord_label = f"({normalized_x:.3f}, {normalized_y:.3f})"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            coord_size = cv2.getTextSize(coord_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # ラベル背景を描画
            cv2.rectangle(frame, (x1, y1-label_size[1]-coord_size[1]-20), 
                        (x1+max(label_size[0], coord_size[0])+10, y1), color, -1)
            
            # ラベルテキストを描画
            cv2.putText(frame, label, (x1+5, y1-coord_size[1]-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, coord_label, (x1+5, y1-5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # UnityにUDP送信
    if persons_data:
        try:
            # JSONデータとして送信
            json_data = json.dumps({
                "timestamp": time.time(),
                "frame_width": frame_width,
                "frame_height": frame_height,
                "persons": persons_data
            })
            sock.sendto(json_data.encode('utf-8'), (UDP_IP, UDP_PORT))
        except Exception as e:
            print(f"UDP送信エラー: {e}")
    
    # 操作説明を表示
    cv2.putText(frame, "Press ESC to exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Detected: {len(persons_data)} people", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"UDP: {UDP_IP}:{UDP_PORT}", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    ellipse_region.draw(frame)
    
    cv2.imshow("Person Tracking with UDP", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESCで終了
        break

cap.release()
cv2.destroyAllWindows()
sock.close()
print("UDP通信を終了しました")