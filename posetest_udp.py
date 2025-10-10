import cv2
import mediapipe as mp
import json
import math
import socket
import time
import random

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Unity通信設定（UDP版）
UNITY_HOST = "127.0.0.1"
UNITY_PORT = 5006  # UDP用のポート
udp_socket = None

def init_udp_communication():
    """UDP通信を初期化"""
    global udp_socket
    try:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP通信を初期化しました ({UNITY_HOST}:{UNITY_PORT})")
        return True
    except Exception as e:
        print(f"UDP初期化失敗: {e}")
        return False

def send_to_unity_udp(message):
    """UnityにUDPメッセージを送信"""
    global udp_socket
    if not udp_socket:
        return False
    
    try:
        json_message = json.dumps(message)
        udp_socket.sendto(json_message.encode('utf-8'), (UNITY_HOST, UNITY_PORT))
        return True
    except Exception as e:
        print(f"UDP送信エラー: {e}")
        return False

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
        pass
    
    return angles

# === サンプルポーズ（あらかじめ保存してあるJSONを読み込み） ===
with open("sample_pose.json", "r") as f:
    all_poses = json.load(f)

print(f"読み込んだポーズ数: {len(all_poses)}")

# 全ポーズの角度情報をリスト化
all_poses_angles = []
for i, pose_data in enumerate(all_poses):
    if 'angles' not in pose_data:
        raise ValueError(f"ポーズ #{i+1} に角度情報が含まれていません。posegen.pyで新しいポーズを作成してください。")
    all_poses_angles.append(pose_data['angles'])

def calc_angle_similarity(current_angles, target_angles):
    """角度の類似度を計算"""
    if not target_angles or not current_angles:
        return float('inf')
    
    errors = []
    for joint in target_angles:
        if joint in current_angles:
            angle_diff = abs(target_angles[joint] - current_angles[joint])
            # 角度の差を0-180度の範囲に正規化
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            errors.append(angle_diff)
    
    return sum(errors) / len(errors) if errors else float('inf')

def check_all_pose_matches(current_angles, threshold=10.0):
    """現在のポーズが全ポーズの中でどれと合致しているかチェック"""
    matched_poses = []
    
    for i, target_angles in enumerate(all_poses_angles):
        similarity = calc_angle_similarity(current_angles, target_angles)
        if similarity < threshold:
            matched_poses.append({
                "pose_number": i + 1,  # 1始まり
                "angle_error": round(similarity, 2)
            })
    
    return matched_poses

# UDP通信を初期化
init_udp_communication()

cap = cv2.VideoCapture(1)

with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # UDP接続状態を表示
        cv2.putText(image, "Unity: UDP Ready", (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if results.pose_landmarks:
            # プレイヤーのポーズ描画
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 現在のポーズの角度を計算
            current_angles = extract_pose_angles(results.pose_landmarks.landmark)

            # 全ポーズとの整合性をチェック
            matched_poses = check_all_pose_matches(current_angles, threshold=15.0)
            
            current_time = time.time()
            
            # 合致しているポーズ情報を表示
            if matched_poses:
                match_text = f"Matched Poses: {', '.join([str(p['pose_number']) for p in matched_poses])}"
                cv2.putText(image, match_text, (50, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # 最も類似度が高いポーズの情報を表示
                best_match = min(matched_poses, key=lambda x: x['angle_error'])
                cv2.putText(image, f"Best Match: Pose #{best_match['pose_number']} (Error: {best_match['angle_error']:.1f}deg)", 
                            (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.putText(image, "POSE MATCHED!", (50, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                cv2.putText(image, "No Match", (50, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(image, "TRY AGAIN", (50, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Unityに合致ポーズ情報を送信
            unity_message = {
                "type": "pose_matches",
                "matched_poses": matched_poses,
                "timestamp": current_time,
                "angles": current_angles
            }
            send_to_unity_udp(unity_message)
            
            # 現在の主要角度を表示
            y_offset = 140
            for joint, angle in current_angles.items():
                cv2.putText(image, f"{joint}: {angle:.1f}deg", (50, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 20
        else:
            cv2.putText(image, "No pose detected", (50, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Pose Game", image)
        
        # キー入力処理
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC
            break

# 終了処理
if udp_socket:
    udp_socket.close()
cap.release()
cv2.destroyAllWindows()
