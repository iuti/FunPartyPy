import cv2
import mediapipe as mp
import json
import os
import math
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

SAVE_FILE = "sample_pose.json"
IMAGE_SIZE = 512  # お手本画像のサイズ（正方形）
IMAGE_MARGIN_RATIO = 0.08  # 画像周囲の余白比率
POSE_IMAGES_DIR = "pose_images"  # ポーズ画像専用ディレクトリ

# ディレクトリが存在しない場合は作成
if not os.path.exists(POSE_IMAGES_DIR):
    os.makedirs(POSE_IMAGES_DIR)

def normalize_coordinates(landmarks):
    """座標を正規化（0-1の範囲に）"""
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    
    # バウンディングボックスを計算
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # アスペクト比を保持しながら中央に配置
    width = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height)
    if max_dim == 0:
        max_dim = 1e-6
    
    # 正方形にフィットするように正規化
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    scale = max(1e-6, 1.0 - 2.0 * IMAGE_MARGIN_RATIO)

    normalized_landmarks = []
    for lm in landmarks:
        # 中央を基準に正規化し、余白分だけスケール
        norm_x = ((lm.x - center_x) / max_dim) * scale + 0.5
        norm_y = ((lm.y - center_y) / max_dim) * scale + 0.5

        # 余白内に収まるようクリップ
        norm_x = float(np.clip(norm_x, IMAGE_MARGIN_RATIO, 1.0 - IMAGE_MARGIN_RATIO))
        norm_y = float(np.clip(norm_y, IMAGE_MARGIN_RATIO, 1.0 - IMAGE_MARGIN_RATIO))
        
        # 画像サイズに合わせてスケール
        pixel_x = int(norm_x * IMAGE_SIZE)
        pixel_y = int(norm_y * IMAGE_SIZE)
        
        normalized_landmarks.append((pixel_x, pixel_y))
    
    return normalized_landmarks

def draw_pose_skeleton(image, landmarks):
    """骨格を描画"""
    # MediaPipeの接続定義
    connections = [
        # 頭部
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        # 胴体
        (9, 10),
        (11, 12), (11, 23), (12, 24), (23, 24),
        # 左腕
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        # 右腕
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        # 左脚
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
        # 右脚
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
    ]
    
    # 接続線を描画
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            cv2.line(image, start_point, end_point, (0, 255, 0), 3)  # 緑色の線
    
    # 関節点を描画
    for point in landmarks:
        cv2.circle(image, point, 5, (255, 255, 255), -1)  # 白い点

def create_pose_reference_image(landmarks, pose_count):
    """お手本となるポーズ画像を生成"""
    # 黒い正方形の背景を作成
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    
    # 座標を正規化
    normalized_landmarks = normalize_coordinates(landmarks)
    
    # 人体の輪郭線を描画（骨格の下に）
    draw_body_outline(image, normalized_landmarks)
    
    # 骨格を描画
    draw_pose_skeleton(image, normalized_landmarks)
    
    # ポーズ番号を表示
    cv2.putText(image, f"Pose #{pose_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image

def calculate_angle(a, b, c):
    """3つの点から角度を計算する関数 (a-b-c の角度)"""
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def extract_pose_angles(landmarks):
    """ポーズから主要な関節角度を抽出"""
    angles = {}
    
    # MediaPipeのランドマークインデックス
    try:
        # 左肘の角度 (肩-肘-手首)
        if all([landmarks[11], landmarks[13], landmarks[15]]):
            angles['left_elbow'] = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        
        # 右肘の角度 (肩-肘-手首)
        if all([landmarks[12], landmarks[14], landmarks[16]]):
            angles['right_elbow'] = calculate_angle(landmarks[12], landmarks[14], landmarks[16])
        
        # 左肩の角度 (肘-肩-腰)
        if all([landmarks[13], landmarks[11], landmarks[23]]):
            angles['left_shoulder'] = calculate_angle(landmarks[13], landmarks[11], landmarks[23])
        
        # 右肩の角度 (肘-肩-腰)
        if all([landmarks[14], landmarks[12], landmarks[24]]):
            angles['right_shoulder'] = calculate_angle(landmarks[14], landmarks[12], landmarks[24])
        
        # 左膝の角度 (腰-膝-足首)
        if all([landmarks[23], landmarks[25], landmarks[27]]):
            angles['left_knee'] = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        
        # 右膝の角度 (腰-膝-足首)
        if all([landmarks[24], landmarks[26], landmarks[28]]):
            angles['right_knee'] = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        
        # 左腰の角度 (肩-腰-膝)
        if all([landmarks[11], landmarks[23], landmarks[25]]):
            angles['left_hip'] = calculate_angle(landmarks[11], landmarks[23], landmarks[25])
        
        # 右腰の角度 (肩-腰-膝)
        if all([landmarks[12], landmarks[24], landmarks[26]]):
            angles['right_hip'] = calculate_angle(landmarks[12], landmarks[24], landmarks[26])
            
    except IndexError:
        print("一部のランドマークが検出されませんでした")
    
    return angles



def _get_landmark_point(landmarks, index):
    if index < len(landmarks):
        x, y = landmarks[index]
        return np.array([x, y], dtype=np.float32)
    return None


def _get_first_valid_point(landmarks, *indices):
    for idx in indices:
        point = _get_landmark_point(landmarks, idx)
        if point is not None:
            return point
    return None


def _draw_limb_polygon(overlay, landmarks, start_idx, end_idx, thickness, color):
    start = _get_landmark_point(landmarks, start_idx)
    end = _get_landmark_point(landmarks, end_idx)
    if start is None or end is None:
        return

    vector = end - start
    length = np.linalg.norm(vector)
    if length < 1e-3:
        return

    direction = vector / length
    perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)
    offset = perpendicular * (thickness / 2.0)

    polygon = np.array([
        start + offset,
        start - offset,
        end - offset,
        end + offset
    ], dtype=np.float32)

    polygon = np.round(polygon).astype(np.int32)
    cv2.fillConvexPoly(overlay, polygon, color)


def draw_body_outline(image, landmarks):
    """簡易的な人体の輪郭線を描画"""
    try:
        if not landmarks:
            return

        overlay = np.zeros_like(image)
        limb_color = (90, 90, 90)
        torso_color = (75, 75, 75)
        head_color = (110, 110, 110)

        shoulder_left = _get_landmark_point(landmarks, 11)
        shoulder_right = _get_landmark_point(landmarks, 12)
        hip_left = _get_landmark_point(landmarks, 23)
        hip_right = _get_landmark_point(landmarks, 24)

        if shoulder_left is not None and shoulder_right is not None:
            body_scale = np.linalg.norm(shoulder_left - shoulder_right)
        elif hip_left is not None and hip_right is not None:
            body_scale = np.linalg.norm(hip_left - hip_right)
        else:
            body_scale = 80.0

        body_scale = max(body_scale, 60.0)

        torso_points = [shoulder_left, shoulder_right, hip_right, hip_left]
        if all(point is not None for point in torso_points):
            hip_extension = body_scale * 0.55
            torso_polygon = np.array([
                shoulder_left,
                shoulder_right,
                hip_right + np.array([body_scale * 0.22, hip_extension], dtype=np.float32),
                hip_left + np.array([-body_scale * 0.22, hip_extension], dtype=np.float32),
                hip_left
            ], dtype=np.float32)
            torso_polygon = np.round(torso_polygon).astype(np.int32)
            cv2.fillConvexPoly(overlay, torso_polygon, torso_color)

        nose = _get_landmark_point(landmarks, 0)
        left_ear = _get_first_valid_point(landmarks, 7, 5)
        right_ear = _get_first_valid_point(landmarks, 8, 6)
        if nose is not None:
            if left_ear is not None and right_ear is not None:
                head_width = np.linalg.norm(left_ear - right_ear)
            else:
                head_width = body_scale * 0.6
            radius_x = max(12, int(head_width * 0.35))
            radius_y = max(14, int(radius_x * 1.3))
            head_center = (int(nose[0]), int(nose[1] - radius_y * 0.25))
            cv2.ellipse(overlay, head_center, (radius_x, radius_y), 0, 0, 360, head_color, -1)

        upper_arm_thickness = max(10, int(body_scale * 0.34))
        forearm_thickness = max(8, int(body_scale * 0.27))
        thigh_thickness = max(14, int(body_scale * 0.44))
        calf_thickness = max(12, int(body_scale * 0.36))

        _draw_limb_polygon(overlay, landmarks, 11, 13, upper_arm_thickness, limb_color)
        _draw_limb_polygon(overlay, landmarks, 13, 15, forearm_thickness, limb_color)
        _draw_limb_polygon(overlay, landmarks, 12, 14, upper_arm_thickness, limb_color)
        _draw_limb_polygon(overlay, landmarks, 14, 16, forearm_thickness, limb_color)
        _draw_limb_polygon(overlay, landmarks, 23, 25, thigh_thickness, limb_color)
        _draw_limb_polygon(overlay, landmarks, 25, 27, calf_thickness, limb_color)
        _draw_limb_polygon(overlay, landmarks, 24, 26, thigh_thickness, limb_color)
        _draw_limb_polygon(overlay, landmarks, 26, 28, calf_thickness, limb_color)

        if shoulder_left is not None and shoulder_right is not None and hip_left is not None and hip_right is not None:
            shoulder_mid = (shoulder_left + shoulder_right) / 2
            hip_mid = (hip_left + hip_right) / 2
            spine_polygon = np.array([
                shoulder_mid + np.array([-body_scale * 0.12, body_scale * 0.02], dtype=np.float32),
                shoulder_mid + np.array([body_scale * 0.12, body_scale * 0.02], dtype=np.float32),
                hip_mid + np.array([body_scale * 0.18, body_scale * 0.55], dtype=np.float32),
                hip_mid + np.array([-body_scale * 0.18, body_scale * 0.55], dtype=np.float32)
            ], dtype=np.float32)
            spine_polygon = np.round(spine_polygon).astype(np.int32)
            cv2.fillConvexPoly(overlay, spine_polygon, (65, 65, 65))

        overlay = cv2.GaussianBlur(overlay, (5, 5), 0)
        cv2.addWeighted(overlay, 0.92, image, 0.08, 0, dst=image)

    except (IndexError, AttributeError, cv2.error) as exc:
        print(f"輪郭線の描画でエラーが発生しました: {exc}")


def save_pose(landmarks, save_file=SAVE_FILE):
    # 座標情報と角度情報の両方を保存
    pose_dict = {
        'coordinates': {i: (lm.x, lm.y) for i, lm in enumerate(landmarks)},
        'angles': extract_pose_angles(landmarks)
    }

    if os.path.exists(save_file):
        with open(save_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(pose_dict)
    pose_count = len(data)

    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    reference_image = create_pose_reference_image(landmarks, pose_count)
    image_filename = os.path.join(POSE_IMAGES_DIR, f"pose_reference_{pose_count:03d}.png")
    cv2.imwrite(image_filename, reference_image)

    print(f"ポーズを {save_file} に保存したよ！（合計 {pose_count} 件）")
    print(f"お手本画像を {image_filename} に保存しました！")
    print(f"保存された角度: {pose_dict['angles']}")

    try:
        cv2.imshow("Pose Reference", reference_image)
        cv2.waitKey(1500)
        cv2.destroyWindow("Pose Reference")
    except cv2.error:
        # GUIが利用できない環境では無視
        pass


# ---- カメラキャプチャ処理 ----
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(image_bgr, "Press SPACE/ENTER to save pose & reference, ESC to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Pose Capture", image_bgr)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (32, 13) and results.pose_landmarks:  # SPACE or ENTER
            save_pose(results.pose_landmarks.landmark)

cap.release()
cv2.destroyAllWindows()
