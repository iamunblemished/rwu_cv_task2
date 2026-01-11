from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import linear_sum_assignment
import pandas as pd

def hungarian_algorithm(cmtr):
    row_ind, col_ind = linear_sum_assignment(-cmtr)
    mask = cmtr[row_ind, col_ind] >= 0.5
    filtered_row_ind = row_ind[mask]
    filtered_col_ind = col_ind[mask]
    
    return filtered_row_ind, filtered_col_ind


img_paths = []
label_paths = []

img_dir = "KITTI_Selection/images/"
label_dir = "KITTI_Selection/labels/"
for filename in os.listdir(img_dir):
    if filename.endswith(".png"):
        img_paths.append(os.path.join(img_dir, filename))
        label_paths.append(os.path.join(label_dir, filename.replace(".png", ".txt")))

model = YOLO("yolo11n.pt") 

index = 0
for img_path, label_path in zip(img_paths, label_paths):  
    gt_boxes = []
    det_boxes = []
    TP = 0
    FP = 0
    FN = 0
    Precision = 0
    Recall = 0
    index = img_path.split("/")[-1].split(".")[0]

    canvas = cv2.imread(img_path)
    
    with open(label_path, 'r') as f:
        for gt_idx, line in enumerate(f):
            parts = line.strip().split(' ')
            if not parts or len(parts) < 6:
                continue
                
            coords = [int(float(x)) for x in parts[1:5]]   
            gt_boxes.append(coords) 
            dist_gt = float(parts[5])
            cv2.rectangle(canvas, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
                
    results = model(img_path, verbose=False)

    for result in results:
        for dt_idx, box in enumerate(result.boxes):
            print(f"BOX: {box}")
            class_id = int(box.cls[0])
            
            if class_id == 2: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det_boxes.append([x1, y1, x2, y2])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)


    # Assuming you have lists: gt_boxes and dt_boxes
    if len(gt_boxes) == 0 or len(det_boxes) == 0:
        # 1. Handle Metrics for empty cases
        TP = 0
        FP = len(det_boxes) # Every detection is a False Positive if there are no GTs
        FN = len(gt_boxes) # Every GT is a False Negative if YOLO detected nothing

        # 2. Skip the Hungarian Algorithm and Matrix logic
        match_GT, match_DT = [], [] 
        Ious = np.array([]) 
        
        print(f"Skipping matching for scene: {index} (GT: {len(gt_boxes)}, DT: {len(det_boxes)})")

    else:
        Ious = []
        for gt in gt_boxes:
            gt_x1, gt_y1, gt_x2, gt_y2 = gt
            Iou_row = []
            for det in det_boxes:
                det_x1, det_y1, det_x2, det_y2 = det

                inter_x1 = max(gt_x1, det_x1)
                inter_y1 = max(gt_y1, det_y1)
                inter_x2 = min(gt_x2, det_x2)
                inter_y2 = min(gt_y2, det_y2)
                
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                
                gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
                det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
                union_area = gt_area + det_area - inter_area
                
                iou = inter_area / union_area
                Iou_row.append(iou)
            Ious.append(Iou_row)

        Ious = np.array(Ious)
        match_GT, match_DT = hungarian_algorithm(Ious)
        num_gt, num_det = Ious.shape
        gt_ind = np.arange(num_gt)
        det_ind = np.arange(num_det)

        TP = len(gt_ind)
        FN = len(gt_ind) - len(match_GT)
        FP = len(det_ind) - len(match_DT)

    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    eval_data = pd.DataFrame()

    eval_data['Metric'] = ['True Positives', 'False Negatives', 'False Positives', 'Precision', 'Recall']
    eval_data['Value'] = [TP, FN, FP, Precision, Recall]

    iou_df = pd.DataFrame()

    results_list = []

    for gt_idx, dt_idx in zip(match_GT, match_DT):
        results_list.append({
            'Ground Truth Coordinates': gt_boxes[gt_idx],
            'Detection Coordinates': det_boxes[dt_idx],
            'IoU Value': Ious[gt_idx, dt_idx]
        })
        x1, y1, x2, y2 = gt_boxes[gt_idx]
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        cv2.putText(canvas, f"{Ious[gt_idx, dt_idx]:.2f}", (x_center - 15, y_center + 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    iou_df = pd.DataFrame(results_list)

    cv2.imwrite(f'GT_YOLO/scene_result_{index}.jpg', canvas)
    eval_data.to_csv(f'Evaluation/eval_scene_{index}.csv', index=False)
    iou_df.to_csv(f'Evaluation/iou_scene_{index}.csv', index=False)

print("\nAll images processed.")
print("Results saved in 'GT_YOLO' and 'Evaluation' folders.")
