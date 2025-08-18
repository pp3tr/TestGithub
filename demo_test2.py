import cv2
import time
import torch
from ultralytics import YOLO
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
import math

# Function to process video with specified model
def process_video(model_path, video_path):
    # Ensure GPU usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the model and move it to the device
    model = YOLO(model_path).to(device)

    # Open video using OpenCV
    cap = cv2.VideoCapture(video_path)

    # Initialize variables
    output_width = 640
    yolo_width = 320  # YOLO model input width
    rotation_angle = 0  ## TODO REMOVE THAT IF THE IMAGE IS GOOD 
    # Dictionaries for static detection and previous bounding boxes for class 0 (boxes)
    static_start_time = {}  # When a box started being static
    class_0_prev_bbox = {}  # Previous bbox for each box (in YOLO coordinate space)
    class_1_count_map = {}
    printed_class_1_counts = set()
    time_threshold = 1  # 1 second of static condition before decision is made
    fps = 0
    frame_count = 0
    overall_start_time = time.time()

    # Tracking variables for boxes and bottles
    class_0_with_12_class_1 = 0
    class_0_without_12_class_1 = 0
    total_class_1_found = 0

    # Dictionaries to map YOLO IDs to new IDs
    class_0_id_mapping = {}
    class_1_id_mapping = {}
    class_0_id_counter = 0
    class_1_id_counter = 0

    # Dictionary to store bottle bounding boxes per box (in YOLO coordinates)
    box_bottle_boxes = {}
    # Dictionary to store the fixed dynamic region per box once decision is made
    box_dynamic_region = {}
    # NEW: Dictionary to store the timestamp when the decision was made for each box
    mesh_decision_time = {}

    # Lists for CSV output
    bottle_data = []
    report_data = []

    # Main loop for video processing
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate frame
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated_frame = cv2.warpAffine(frame, M, (width, height))

        # Create two resized versions:
        # 1. For display (output) at desired output_width.
        # 2. For YOLO input at yolo_width (320 p).
        aspect_ratio = height / width
        resized_frame = cv2.resize(rotated_frame, (output_width, int(output_width * aspect_ratio)))
        yolo_frame = cv2.resize(rotated_frame, (yolo_width, int(yolo_width * aspect_ratio)))
        # Scale factor to map YOLO coordinates to output image coordinates
        scale_factor = output_width / yolo_width

        # Perform tracking on the YOLO input frame (320 p)
        results = model.track(yolo_frame, conf=0.7, classes=[0, 1], persist=True, verbose=False)

        detected_class_0 = []
        detected_class_1 = []

        # Process detections from YOLO results
        for result in results:
            detections = result.boxes

            for det in detections:
                class_id = int(det.cls)

                # Check for None in det.id before converting
                if det.id is None:
                    if class_id == 1:
                        new_obj_id = None
                    else:
                        continue
                else:
                    obj_id = int(det.id)

                # Get the bounding box (from YOLO frame)
                bbox = det.xyxy[0].tolist()  # [x1, y1, x2, y2]

                if class_id == 0:  # Box detected
                    if det.id is not None and obj_id not in class_0_id_mapping:
                        class_0_id_mapping[obj_id] = class_0_id_counter
                        class_0_id_counter += 1
                    if det.id is not None:
                        new_obj_id = class_0_id_mapping[obj_id]
                    detected_class_0.append(new_obj_id)

                    # Compare current bbox with previous bbox to determine static state
                    if new_obj_id in class_0_prev_bbox:
                        prev_bbox = class_0_prev_bbox[new_obj_id]
                        prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)
                        curr_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                        distance = math.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])
                        if distance <= 5:
                            if new_obj_id not in static_start_time:
                                static_start_time[new_obj_id] = time.time()
                        else:
                            if new_obj_id in static_start_time:
                                del static_start_time[new_obj_id]
                    class_0_prev_bbox[new_obj_id] = bbox

                    # If the box has been static for at least time_threshold, count the bottles inside it
                    if new_obj_id in static_start_time and (time.time() - static_start_time[new_obj_id] >= time_threshold):
                        class_1_count = 0
                        bottle_boxes = []  # To store bottle bounding boxes for mesh overlay
                        for det_inner in detections:
                            if int(det_inner.cls) == 1:  # Bottle detected
                                inner_bbox = det_inner.xyxy[0].tolist()
                                # Check if the bottle bbox overlaps with the box bbox
                                if not (inner_bbox[2] < bbox[0] or inner_bbox[0] > bbox[2] or
                                        inner_bbox[3] < bbox[1] or inner_bbox[1] > bbox[3]):
                                    class_1_count += 1
                                    bottle_boxes.append(inner_bbox)
                        if new_obj_id not in printed_class_1_counts:
                            print(f"Bottles count within Box with ID {new_obj_id} (static for 1s): {class_1_count}")
                            class_1_count_map[new_obj_id] = class_1_count
                            printed_class_1_counts.add(new_obj_id)
                            time_csv = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            bottle_data.append([time_csv, class_1_count, new_obj_id])
                            if class_1_count == 12:
                                class_0_with_12_class_1 += 1
                            else:
                                class_0_without_12_class_1 += 1
                            total_class_1_found += class_1_count
                            if class_1_count < 12:
                                print("Warning: fewer bottles detected")
                            # Store bottle bounding boxes for mesh overlay
                            box_bottle_boxes[new_obj_id] = bottle_boxes
                            # Compute and store the fixed dynamic region based on the outer bottle bboxes
                            if bottle_boxes:
                                region_x1 = min(b[0] for b in bottle_boxes)
                                region_y1 = min(b[1] for b in bottle_boxes)
                                region_x2 = max(b[2] for b in bottle_boxes)
                                region_y2 = max(b[3] for b in bottle_boxes)
                                margin_factor = 0.1  # 10% margin
                                region_width = region_x2 - region_x1
                                region_height = region_y2 - region_y1
                                margin_x = margin_factor * region_width
                                margin_y = margin_factor * region_height
                                dynamic_x1 = max(bbox[0], region_x1 - margin_x)
                                dynamic_y1 = max(bbox[1], region_y1 - margin_y)
                                dynamic_x2 = min(bbox[2], region_x2 + margin_x)
                                dynamic_y2 = min(bbox[3], region_y2 + margin_y)
                            else:
                                dynamic_x1, dynamic_y1, dynamic_x2, dynamic_y2 = bbox
                            box_dynamic_region[new_obj_id] = [dynamic_x1, dynamic_y1, dynamic_x2, dynamic_y2]
                            # NEW: Record the decision time for this box so the mesh is shown only for 0.5 sec
                            mesh_decision_time[new_obj_id] = time.time()

                elif class_id == 1:  # Bottle detected
                    if det.id is not None:
                        if obj_id not in class_1_id_mapping:
                            class_1_id_mapping[obj_id] = class_1_id_counter
                            class_1_id_counter += 1
                        new_obj_id = class_1_id_mapping[obj_id]
                    detected_class_1.append(new_obj_id)

        # Remove boxes no longer detected from static tracking and previous bbox storage
        static_start_time = {k: v for k, v in static_start_time.items() if k in detected_class_0}
        class_0_prev_bbox = {k: v for k, v in class_0_prev_bbox.items() if k in detected_class_0}
        # Also clear dynamic regions for boxes that are no longer static
        for box_id in list(box_dynamic_region.keys()):
            if box_id not in static_start_time:
                del box_dynamic_region[box_id]
        # And clear mesh decision times for boxes no longer static
        for box_id in list(mesh_decision_time.keys()):
            if box_id not in static_start_time:
                del mesh_decision_time[box_id]

        # Annotate the output frame with bounding boxes and labels (scaled to output size)
        colors = {0: (0, 255, 0), 1: (0, 0, 255)}
        if results:
            detections = results[-1].boxes
            for det in detections:
                class_id = int(det.cls)
                x1, y1, x2, y2 = det.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(y2 * scale_factor)
                if class_id == 0:
                    if det.id is not None:
                        obj_id = int(det.id)
                        new_obj_id = class_0_id_mapping.get(obj_id, None)
                        label = f"ID: {new_obj_id}" if new_obj_id is not None else "Box"
                    else:
                        label = "Box"
                elif class_id == 1:
                    label = "Bottle"
                else:
                    label = "Object"
                color = colors.get(class_id, (255, 255, 255))
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(resized_frame, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        #         # ---- New Code: Draw fixed dynamic mesh overlay for 0.5 sec after decision ----
        grid_cols = 3
        grid_rows = 4
        current_time = time.time()
        # Iterate over boxes with a stored dynamic region (i.e. decision made and box still static)
        for box_id, dynamic_region in box_dynamic_region.items():
            # Only show mesh if the decision was made within the last 0.5 seconds
            if box_id not in mesh_decision_time or (current_time - mesh_decision_time[box_id]) > 0.5:
                continue

            # Use the stored dynamic region (in YOLO coordinates)
            dynamic_x1, dynamic_y1, dynamic_x2, dynamic_y2 = dynamic_region
            # Scale the dynamic region to output frame coordinates
            scaled_dynamic = [int(coord * scale_factor) for coord in [dynamic_x1, dynamic_y1, dynamic_x2, dynamic_y2]]
            x1_dyn, y1_dyn, x2_dyn, y2_dyn = scaled_dynamic
            cell_width = (x2_dyn - x1_dyn) / grid_cols
            cell_height = (y2_dyn - y1_dyn) / grid_rows

            # For each cell in the fixed grid, evaluate overlap using maximum ratio over all bottle bboxes
            for row in range(grid_rows):
                for col in range(grid_cols):
                    cell_x1 = int(x1_dyn + col * cell_width)
                    cell_y1 = int(y1_dyn + row * cell_height)
                    cell_x2 = int(x1_dyn + (col + 1) * cell_width)
                    cell_y2 = int(y1_dyn + (row + 1) * cell_height)
                    cell_area = (cell_x2 - cell_x1) * (cell_y2 - cell_y1)
                    max_overlap_ratio = 0.0
                    # Evaluate each bottle bbox from the stored bottle data
                    for bbox_bottle in box_bottle_boxes.get(box_id, []):
                        bx1, by1, bx2, by2 = [x * scale_factor for x in bbox_bottle]
                        inter_x1 = max(cell_x1, bx1)
                        inter_y1 = max(cell_y1, by1)
                        inter_x2 = min(cell_x2, bx2)
                        inter_y2 = min(cell_y2, by2)
                        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                            overlap_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                            overlap_ratio = overlap_area / cell_area
                            if overlap_ratio > max_overlap_ratio:
                                max_overlap_ratio = overlap_ratio
                    # Use a threshold to decide cell color (e.g. 0.3)
                    threshold = 0.3
                    cell_color = (0, 255, 0) if max_overlap_ratio >= threshold else (0, 0, 255)
                    overlay = resized_frame.copy()
                    cv2.rectangle(overlay, (cell_x1, cell_y1), (cell_x2, cell_y2), cell_color, -1)
                    alpha = 0.8
                    resized_frame = cv2.addWeighted(overlay, alpha, resized_frame, 1 - alpha, 0)
                    cv2.rectangle(resized_frame, (cell_x1, cell_y1), (cell_x2, cell_y2), (255, 255, 255), 1)

                    # Draw checkmark (✓) on green cells
                    center_x = (cell_x1 + cell_x2) // 2
                    center_y = (cell_y1 + cell_y2) // 2
                    if max_overlap_ratio >= threshold:
                        checkmark_start = (center_x - 5, center_y + 5)
                        checkmark_mid = (center_x, center_y + 10)
                        checkmark_end = (center_x + 10, center_y - 5)
                        cv2.line(resized_frame, checkmark_start, checkmark_mid, (0, 0, 0), 2)  # Left leg of checkmark
                        cv2.line(resized_frame, checkmark_mid, checkmark_end, (0, 0, 0), 2)  # Right leg of checkmark
                    else:
                        # Draw "X" on red cells
                        x_start = (center_x - 7, center_y - 7)
                        x_end = (center_x + 7, center_y + 7)
                        cv2.line(resized_frame, x_start, x_end, (255, 255, 255), 2)
                        x_start = (center_x - 7, center_y + 7)
                        x_end = (center_x + 7, center_y - 7)
                        cv2.line(resized_frame, x_start, x_end, (255, 255, 255), 2)

        # # --------------------------------------------------

        # Calculate and display FPS
        frame_count += 1
        if time.time() - overall_start_time >= 1.0:
            fps = frame_count
            frame_count = 0
            overall_start_time = time.time()

        cv2.putText(resized_frame, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('YOLO Tracking', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    #### Post-processing: Calculate expected bottles and generate reports
    print("Calculating total expected bottles...")
    expected_class_1_total = len(class_1_count_map) * 12
    print(f"Expected total of Bottles: {expected_class_1_total}")

    total_class_1_found = sum(class_1_count_map.values())
    print(f"Total Bottles found so far: {total_class_1_found}")
    wrong_bottle_count = 0

    good_boxes = 0
    bad_boxes = 0
    fewer_than_12_counts = []

    print("Calculating counts for good and bad boxes...")
    for count in class_1_count_map.values():
        print(f"Processing box with count: {count}")
        if count == 12:
            good_boxes += 1
        elif count < 12:
            bad_boxes += 1
            fewer_than_12_counts.append(count)

    print(f"Good boxes: {good_boxes}, Bad boxes: {bad_boxes}")

    total_class_1_found = sum(class_1_count_map.values())
    expected_class_1_total = len(class_1_count_map) * 12
    wrong_bottle_count = expected_class_1_total - total_class_1_found

    print("Creating final report data...")
    report_data.append(["Boxes with exactly 12 Bottles inside", good_boxes])
    report_data.append(["Boxes with fewer than 12 Bottles inside", bad_boxes])
    report_data.append(["Counts of Boxes with fewer than 12 Bottles", fewer_than_12_counts])

    bottles_csv = 'bottles_data.csv'
    print(f"Saving bottle data to {bottles_csv}...")
    with open(bottles_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Bottle Count", "Box ID"])
        writer.writerows(bottle_data)

    report_csv = 'report_data.csv'
    print(f"Saving report data to {report_csv}...")
    with open(report_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(report_data)

model_list = ['Model_m_data_all_last_try']

video_path = r"C:/Users/pp3tr/Desktop/bottle_demos/test_for_demo.mp4"
model_path = r"C:/Users/pp3tr/Desktop/bottle_demos/best.pt"
process_video(model_path, video_path)