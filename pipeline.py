import os
import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from ultralytics import YOLO

# === [第一階段] 使用YOLO模型推論找出joint區域並進行模糊與裁切 ===
# 設定路徑
input_folder = "/home/yl732/r12945059/yolov10/2024032613382100220240326133821002"
model_path = "/home/yl732/r12945059/yolov10/runs/detect/train/weights/best.pt"  # 可替換YOLO模型路徑
output_folder = os.path.join(input_folder, "Gaussian_joint_box_output")
cropped_folder = os.path.join(output_folder, "cropped")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(cropped_folder, exist_ok=True)

# 載入YOLO模型
model = YOLO(model_path)

image_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
joint_boxes = {}  # 儲存每張圖的 joint box 區域

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # 使用YOLO模型進行推論
    results = model(image)
    
    joint_box = None
    
    # 處理推論結果
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                # 獲取類別ID和信心度
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 檢查是否為joint類別(假設joint的類別ID為3)且信心度足夠
                if cls == 3 and conf > 0.5:  # 可調整信心度閾值
                    # 獲取邊界框座標 (xyxy格式)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    joint_box = (x1, y1, x2, y2)
                    joint_boxes[image_file] = joint_box
                    break  # 找到第一個joint就停止
    
    # 如果沒找到joint，直接保存原圖
    if joint_box is None:
        print(f"在 {image_file} 中未檢測到joint或信心度不足")
        cv2.imwrite(os.path.join(output_folder, image_file), image)
        continue

    # 處理找到joint的情況
    x1, y1, x2, y2 = joint_box
    
    # 確保座標在圖像範圍內
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # 提取ROI並進行高斯模糊
    roi = image[y1:y2, x1:x2]
    if roi.size > 0:  # 確保ROI不為空
        blurred_roi = cv2.GaussianBlur(roi, (9, 9), 0)
        image[y1:y2, x1:x2] = blurred_roi
        cv2.imwrite(os.path.join(cropped_folder, image_file), roi)
        print(f"成功處理 {image_file}，joint區域: ({x1}, {y1}, {x2}, {y2})")
    else:
        print(f"警告：{image_file} 的joint區域為空")
    
    cv2.imwrite(os.path.join(output_folder, image_file), image)

print(f"第一階段完成，共處理 {len(joint_boxes)} 張包含joint的圖片")

# === [第二階段] 對裁切圖做 adaptive threshold 處理 ===
adaptive_input_dir = cropped_folder
adaptive_output_dir = os.path.join(output_folder, "adaptive_threshold")
os.makedirs(adaptive_output_dir, exist_ok=True)

processed_count = 0
for filename in os.listdir(adaptive_input_dir):
    input_path = os.path.join(adaptive_input_dir, filename)
    if not filename.lower().endswith(('.png', '.jpg')):
        continue
    
    img = cv2.imread(input_path, cv2.IMREAD_AYSCALGRE)
    if img is None:
        continue
    
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    _, global_thresh = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_and(adaptive_thresh, global_thresh)
    cv2.imwrite(os.path.join(adaptive_output_dir, filename), combined)
    processed_count += 1

print(f"第二階段完成，處理了 {processed_count} 張adaptive threshold圖片")

# === [第三階段] 從二值圖中找出兩端白線與距離，畫回 joint 區域 ===
processed_output_dir = os.path.join(output_folder, "processed")
os.makedirs(processed_output_dir, exist_ok=True)

def find_longest_line(region):
    """找出區域中最長的線段"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(region, connectivity=8)
    max_length = 0
    longest_line_endpoints = None
    
    for label in range(1, num_labels):
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) >= 2:
                # 找出輪廓中距離最遠的兩點
                for i in range(len(contour)):
                    for j in range(i + 1, len(contour)):
                        p1 = contour[i][0]
                        p2 = contour[j][0]
                        distance = euclidean(p1, p2)
                        if distance > max_length:
                            max_length = distance
                            longest_line_endpoints = (tuple(p1), tuple(p2))
    
    return longest_line_endpoints

measurement_count = 0
for filename in os.listdir(adaptive_output_dir):
    input_path = os.path.join(adaptive_output_dir, filename)
    if not filename.lower().endswith('.png'):
        continue

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    # 載入原圖
    original_path = os.path.join(output_folder, filename)
    original_img = cv2.imread(original_path)
    if original_img is None:
        original_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 二值化處理
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    height, width = binary.shape
    
    # 分割左右區域
    left_region = binary[:, :width // 2]
    right_region = binary[:, width // 2:]

    # 找出左右區域的最長線段
    left_endpoints = find_longest_line(left_region)
    right_endpoints = find_longest_line(right_region)

    # 如果找到兩條線段且該圖片有joint box資訊
    if left_endpoints and right_endpoints and filename in joint_boxes:
        joint_x1, joint_y1, joint_x2, joint_y2 = joint_boxes[filename]

        # 計算實際座標點
        left_point = (left_endpoints[1][0] + joint_x1, left_endpoints[1][1] + joint_y1)
        right_point = (right_endpoints[0][0] + width // 2 + joint_x1, right_endpoints[0][1] + joint_y1)

        # 計算距離
        distance = euclidean(left_point, right_point)

        # 繪製標記的輔助函數
        def draw_offset(pt):
            return (pt[0] + joint_x1, pt[1] + joint_y1)

        # 繪製左側線段
        cv2.circle(original_img, draw_offset(left_endpoints[0]), 5, (0, 0, 255), -1)
        cv2.circle(original_img, draw_offset(left_endpoints[1]), 5, (0, 0, 255), -1)
        cv2.line(original_img, draw_offset(left_endpoints[0]), draw_offset(left_endpoints[1]), (0, 0, 255), 2)

        # 繪製右側線段
        cv2.circle(original_img, draw_offset((right_endpoints[0][0] + width // 2, right_endpoints[0][1])), 5, (0, 0, 255), -1)
        cv2.circle(original_img, draw_offset((right_endpoints[1][0] + width // 2, right_endpoints[1][1])), 5, (0, 0, 255), -1)
        cv2.line(original_img, draw_offset((right_endpoints[0][0] + width // 2, right_endpoints[0][1])),
                 draw_offset((right_endpoints[1][0] + width // 2, right_endpoints[1][1])), (0, 0, 255), 2)

        # 繪製測量線
        cv2.line(original_img, left_point, right_point, (255, 0, 0), 2)
        cv2.circle(original_img, left_point, 5, (255, 0, 0), -1)
        cv2.circle(original_img, right_point, 5, (255, 0, 0), -1)

        # 標示距離
        label = f"{distance:.1f} px"
        label_x = joint_x1 + 10
        label_y = joint_y1 + 30
        cv2.putText(original_img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        measurement_count += 1
        print(f"{filename}: joint gap = {distance:.1f} px")

    cv2.imwrite(os.path.join(processed_output_dir, filename), original_img)

print(f"第三階段完成，成功測量了 {measurement_count} 張圖片的joint gap")
print(f"結果保存在: {processed_output_dir}")