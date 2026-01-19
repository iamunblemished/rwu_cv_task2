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

def read_calibration_matrix(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        P2 = None
        for line in lines:
            data = line.strip().split()
            numbers = np.array([float(n) for n in data])
            P2 = np.append(P2, numbers) if P2 is not None else numbers
        P2 = P2.reshape(3, 3)
        return P2

img_paths = []
label_paths = []
calib_paths = []

img_dir = "KITTI_Selection/images/"
label_dir = "KITTI_Selection/labels/"
calib_dir = "KITTI_Selection/calib/"
for filename in os.listdir(img_dir):
    if filename.endswith(".png"):
        img_paths.append(os.path.join(img_dir, filename))
        label_paths.append(os.path.join(label_dir, filename.replace(".png", ".txt")))
        calib_paths.append(os.path.join(calib_dir, filename.replace(".png", ".txt")))

model = YOLO("yolo11n.pt") 

index = 0
for img_path, label_path, calib_path in zip(img_paths, label_paths, calib_paths):  
    gt_boxes = []
    det_boxes = []
    TP = 0
    FP = 0
    FN = 0
    Precision = 0
    Recall = 0
    index = img_path.split("/")[-1].split(".")[0]
    dist_gt = []
    dist_dt = []

    canvas = cv2.imread(img_path)
    kmatrix = read_calibration_matrix(calib_path)
    cx = kmatrix[0, 2]
    cy = kmatrix[1, 2]
    fx = kmatrix[0, 0]
    fy = kmatrix[1, 1]
    
    with open(label_path, 'r') as f:
        for gt_idx, line in enumerate(f):
            parts = line.strip().split(' ')
            if not parts or len(parts) < 6:
                continue
                
            coords = [int(float(x)) for x in parts[1:5]]   
            gt_boxes.append(coords) 
            dist_gt.append(float(parts[5]))
            cv2.rectangle(canvas, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
                
    results = model(img_path, verbose=False)

    for result in results:
        for dt_idx, box in enumerate(result.boxes):
            print(f"BOX: {box}")
            class_id = int(box.cls[0])
            
            if class_id == 2: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det_boxes.append([x1, y1, x2, y2])
                u = (x1 + x2) / 2
                v = y2
                depth = (fy * 1.65) / (v - cy)
                offset = depth * (u - cx) / fx
                dist_dt.append(np.sqrt(depth**2 + offset**2))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for calib_line in open(calib_path):
        if calib_line.startswith("P2:"):
            P2_values = list(map(float, calib_line.strip().split()[1:]))
            P2 = np.array(P2_values).reshape(3, 4)
            break

    if len(gt_boxes) == 0 or len(det_boxes) == 0:
        TP = 0
        FP = len(det_boxes)
        FN = len(gt_boxes)

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

        TP = len(match_GT)
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
            'IoU Value': Ious[gt_idx, dt_idx],
            'Distance GT': dist_gt[gt_idx],
            'Distance DT': dist_dt[dt_idx],
            'Distance Error (m)': abs(dist_gt[gt_idx] - dist_dt[dt_idx]),
        })
        x1, y1, x2, y2 = gt_boxes[gt_idx]
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        cv2.putText(canvas, f"{Ious[gt_idx, dt_idx]:.2f}", (x_center - 15, y_center + 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    iou_df = pd.DataFrame(results_list)

    # plt.figure(figsize=(10, 6))
    # plt.scatter(dist_gt_graph, dist_dt_graph, alpha=0.5)
    # plt.plot([0, max(dist_gt_graph)], [0, max(dist_gt_graph)], color='red', linestyle='--')
    # plt.xlabel('Ground Truth Distance (m)')
    # plt.ylabel('Detected Distance (m)')
    # plt.title('Detected vs Ground Truth Distances')
    # plt.grid()
    # plt.savefig('Evaluation/distance_comparison.png')
    # plt.show()

    cv2.imwrite(f'GT_YOLO/scene_result_{index}.jpg', canvas)
    eval_data.to_csv(f'Evaluation/eval_scene_{index}.csv', index=False)
    iou_df.to_csv(f'Evaluation/iou_scene_{index}.csv', index=False)

print("\nAll images processed.")
print("Results saved in 'GT_YOLO' and 'Evaluation' folders.")
