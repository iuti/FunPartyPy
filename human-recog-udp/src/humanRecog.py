import cv2
import numpy as np
from ultralytics import YOLO
import torch
import socket
import pickle
import struct

class HumanDetector:
    def __init__(self, unity_ip='127.0.0.1', unity_port=9999):
        # デバイス設定
        self.use_cuda = torch.cuda.is_available()
        self.device_str = "cuda:0" if self.use_cuda else "cpu"
        print(f"[HumanDetector] Torch CUDA available: {self.use_cuda}, loading models on {self.device_str}")

        # YOLOv8モデルをロード（人物検出用） - 利用可能ならGPUにロード
        # Load detection model (always load, then move to device if possible)
        try:
            self.model = YOLO('yolov8n.pt')
            if self.use_cuda:
                try:
                    # move underlying torch model to device
                    if hasattr(self.model, 'to'):
                        try:
                            # Some ultralytics versions implement .to
                            self.model.to(self.device_str)
                        except Exception:
                            pass
                    if hasattr(self.model, 'model'):
                        try:
                            self.model.model.to(self.device_str)
                        except Exception:
                            pass
                    print(f"[HumanDetector] Moved detection model to {self.device_str}")
                except Exception as e:
                    print(f"[HumanDetector] Could not move detection model to {self.device_str}: {e}")
        except Exception as e:
            print(f"[HumanDetector] Failed to load detection model: {e}")
            raise

        # セグメンテーション用モデル
        # Load segmentation model and move to device if possible
        try:
            self.seg_model = YOLO('yolov8n-seg.pt')
            if self.use_cuda:
                try:
                    if hasattr(self.seg_model, 'to'):
                        try:
                            self.seg_model.to(self.device_str)
                        except Exception:
                            pass
                    if hasattr(self.seg_model, 'model'):
                        try:
                            self.seg_model.model.to(self.device_str)
                        except Exception:
                            pass
                    print(f"[HumanDetector] Moved segmentation model to {self.device_str}")
                except Exception as e:
                    print(f"[HumanDetector] Could not move segmentation model to {self.device_str}: {e}")
        except Exception as e:
            print(f"[HumanDetector] Failed to load segmentation model: {e}")
            raise

        # NOTE: we do not force half precision conversion here because that
        # can cause dtype mismatches (Half vs Float) during matmul if inputs
        # or some parts of the pipeline remain float32. If you want to use
        # float16, either ensure the entire preprocessing and inputs are
        # converted to float16, or use an API that supports half inference.
        
        # UDP通信設定
        self.unity_ip = unity_ip
        self.unity_port = unity_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        print(f"Unity送信先: {unity_ip}:{unity_port}")
        
    def detect_humans(self, frame):
        """
        フレーム内の人を検出
        """
        results = self.model(frame)
        human_boxes = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # クラス0は人（COCO dataset）
                    if int(box.cls) == 0:  # person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        human_boxes.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence)
                        })
        
        return human_boxes
    
    def remove_background_with_alpha(self, frame, confidence_threshold=0.5):
        """
        人物のセグメンテーションで背景を透過処理
        """
        results = self.seg_model(frame)
        
        # 元画像と同じサイズのマスクを作成
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes
                
                for i, (box_mask, box) in enumerate(zip(masks, boxes)):
                    # 人物クラス（クラス0）かつ信頼度が閾値以上
                    if int(box.cls) == 0 and float(box.conf) >= confidence_threshold:
                        # マスクをリサイズ
                        resized_mask = cv2.resize(box_mask, (frame.shape[1], frame.shape[0]))
                        # マスクを0-255の範囲に変換
                        binary_mask = (resized_mask > 0.5).astype(np.uint8) * 255
                        mask = cv2.bitwise_or(mask, binary_mask)
        
        # BGRAフォーマットで透明度付き画像を作成
        height, width = frame.shape[:2]
        result_frame = np.zeros((height, width, 4), dtype=np.uint8)
        
        # RGB部分をコピー
        result_frame[:, :, :3] = frame[:, :, :3]  # BGR
        
        # アルファチャンネル（透明度）を設定
        result_frame[:, :, 3] = mask  # マスクをアルファチャンネルに
        
        return result_frame, mask
    
    def send_image_to_unity(self, image):
        """
        透明度付き画像をUnityにUDP送信
        """
        try:
            max_payload = 60000  # UDP安全域
            scale = 1.0
            # そのまま送信（左右反転は Unity 側で処理する）
            resized_image = image

            while True:
                success, buffer = cv2.imencode('.png', resized_image)
                if not success:
                    raise ValueError("PNGエンコードに失敗しました")
                data = buffer.tobytes()

                if len(data) <= max_payload:
                    break

                scale *= 0.8
                if scale < 0.1:
                    raise ValueError("画像を十分に縮小できませんでした")

                height, width = resized_image.shape[:2]
                new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                resized_image = cv2.resize(resized_image, new_size, interpolation=cv2.INTER_AREA)

            size = struct.pack("I", len(data))
            self.sock.sendto(size + data, (self.unity_ip, self.unity_port))

            print(f"透明度付き画像を送信: {len(data)} bytes (scale={scale:.2f})")

        except Exception as e:
            print(f"Unity送信エラー: {e}")
    
    def send_cropped_human_to_unity(self, frame):
        """
        人だけくり抜いた透明度付き画像をUnityに送信
        """
        # 背景を透過処理
        bg_removed_frame, mask = self.remove_background_with_alpha(frame)
        
        # マスクから人物領域のバウンディングボックスを取得
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 最大の輪郭を取得（最も大きな人物）
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 人物部分だけをクロップ（透明度込み）
            if w > 50 and h > 50:  # 最小サイズチェック
                # 少し余裕を持ってクロップ（境界を含む）
                padding = 10
                x = x - padding
                y = y - padding
                w = w + padding * 2
                h = h + padding * 2

                # 正方形サイズを決定（人物を収める最小の正方形）
                side = max(w, h)

                # 中心座標
                cx = x + w // 2
                cy = y + h // 2

                # 正方形の左上座標
                left = int(cx - side // 2)
                top = int(cy - side // 2)
                right = left + int(side)
                bottom = top + int(side)

                img_h, img_w = bg_removed_frame.shape[:2]

                # 取得元の範囲（画像内にクリップ）
                src_x1 = max(0, left)
                src_y1 = max(0, top)
                src_x2 = min(img_w, right)
                src_y2 = min(img_h, bottom)

                # 出力キャンバス（アルファ付き）を透過で作成
                out = np.zeros((int(side), int(side), 4), dtype=np.uint8)

                # コピー先のオフセット
                dst_x1 = src_x1 - left
                dst_y1 = src_y1 - top

                # コピーサイズ
                copy_w = src_x2 - src_x1
                copy_h = src_y2 - src_y1

                if copy_w > 0 and copy_h > 0:
                    # 入力が BGRA でない場合は BGRA に変換（安全策）
                    src = bg_removed_frame[src_y1:src_y2, src_x1:src_x2]
                    if src.shape[2] == 3:
                        # RGB -> BGRA (alpha 255)
                        alpha_ch = np.full((src.shape[0], src.shape[1], 1), 255, dtype=np.uint8)
                        src = np.concatenate([src, alpha_ch], axis=2)

                    out[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = src

                # Unityに送信
                self.send_image_to_unity(out)

                return out
        
        return None
    
    def remove_background(self, frame, confidence_threshold=0.5):
        """
        従来の背景除去（表示用）
        """
        results = self.seg_model(frame)
        
        # 元画像と同じサイズのマスクを作成
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes

                for i, (box_mask, box) in enumerate(zip(masks, boxes)):
                    # 人物クラス（クラス0）かつ信頼度が閾値以上
                    if int(box.cls) == 0 and float(box.conf) >= confidence_threshold:
                        # マスクをリサイズ
                        resized_mask = cv2.resize(box_mask, (frame.shape[1], frame.shape[0]))
                        # マスクを0-255の範囲に変換
                        binary_mask = (resized_mask > 0.5).astype(np.uint8) * 255
                        mask = cv2.bitwise_or(mask, binary_mask)

        # 背景除去
        result_frame = frame.copy()
        result_frame[mask == 0] = [0, 0, 0]  # 背景を黒にする

        return result_frame, mask

    def process_frame(self, frame):
        """
        フレームを処理して人物検出と背景除去を行う
        """
        # 人物検出
        humans = self.detect_humans(frame)

        # 検出結果を描画
        detection_frame = frame.copy()
        for human in humans:
            bbox = human['bbox']
            confidence = human['confidence']

            # バウンディングボックスを描画
            cv2.rectangle(detection_frame, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         (0, 255, 0), 2)

            # 信頼度を表示
            cv2.putText(detection_frame, 
                       f'Person: {confidence:.2f}',
                       (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 従来の背景除去（表示用）
        bg_removed_frame, mask = self.remove_background(frame)

        # 人だけくり抜いた透明度付き画像をUnityに送信
        cropped_human = self.send_cropped_human_to_unity(frame)

        return detection_frame, bg_removed_frame, mask, cropped_human

def main():
    # カメラを初期化
    cap = cv2.VideoCapture(0)
    
    # 人物検出器を初期化
    detector = HumanDetector()
    
    print("カメラを開始します。'q'キーで終了、's'キーで画像保存")
    print("透明度付き画像をUnityに送信中...")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # フレームを処理
        detection_frame, bg_removed_frame, mask, cropped_human = detector.process_frame(frame)
        
        # 結果を表示
        cv2.imshow('Original', frame)
        cv2.imshow('Human Detection', detection_frame)
        cv2.imshow('Background Removed', bg_removed_frame)
        cv2.imshow('Mask', mask)
        
        # クロップされた人物画像も表示（透明度は表示されない）
        if cropped_human is not None:
            # 透明度を可視化するため、チェッカーボード背景に合成
            checker_bg = np.ones((cropped_human.shape[0], cropped_human.shape[1], 3), dtype=np.uint8) * 128
            display_img = checker_bg.copy()
            
            if cropped_human.shape[2] == 4:  # アルファチャンネルがある場合
                alpha = cropped_human[:, :, 3:4] / 255.0
                display_img = display_img * (1 - alpha) + cropped_human[:, :, :3] * alpha
                display_img = display_img.astype(np.uint8)
            else:
                display_img = cropped_human[:, :, :3]
            
            cv2.imshow('Transparent Human (To Unity)', display_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 画像を保存
            cv2.imwrite(f'original_{frame_count}.jpg', frame)
            cv2.imwrite(f'detection_{frame_count}.jpg', detection_frame)
            cv2.imwrite(f'bg_removed_{frame_count}.jpg', bg_removed_frame)
            cv2.imwrite(f'mask_{frame_count}.jpg', mask)
            if cropped_human is not None:
                cv2.imwrite(f'transparent_human_{frame_count}.png', cropped_human)
            print(f"フレーム {frame_count} を保存しました")
            frame_count += 1
    
    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

