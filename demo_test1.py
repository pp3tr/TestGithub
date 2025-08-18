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
    output_width = 1280
    yolo_width = 320  # New variable for YOLO input size
    rotation_angle = -8 ## TODO REMOVE THAT IF THE IMAGE IS GOOD 
    # Dictionaries for static detection and previous bounding boxes for class 0 (boxes)
    static_start_time = {}  # When a box started being static
    class_0_prev_bbox = {}  # Previous bbox for each box
    class_1_count_map = {}
    printed_class_1_counts = set()
    time_threshold = 0.75  # 1 second of static condition before decision is made
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

        # Compute aspect ratio and create two resized versions:
        # 1. One for display (output) at your desired output_width.
        # 2. One for YOLO input at yolo_width (320 p).
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
                        # For bottles, assign new_obj_id as None and continue processing
                        new_obj_id = None
                    else:
                        # For boxes, skip if no ID is provided
                        continue
                else:
                    obj_id = int(det.id)

                # Get the original bounding box (from YOLO frame)
                bbox = det.xyxy[0].tolist()

                if class_id == 0:  # Box detected
                    # Map YOLO box ID to a new ID if not already mapped
                    if det.id is not None and obj_id not in class_0_id_mapping:
                        class_0_id_mapping[obj_id] = class_0_id_counter
                        class_0_id_counter += 1

                    # Use mapped ID if available
                    if det.id is not None:
                        new_obj_id = class_0_id_mapping[obj_id]
                    detected_class_0.append(new_obj_id)

                    # Static detection: compare current bbox with previous bbox
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
                    # For a new box, record its bbox for future comparisons
                    class_0_prev_bbox[new_obj_id] = bbox

                    # If the box has been static for at least 1 second, count the bottles inside it
                    if new_obj_id in static_start_time and (time.time() - static_start_time[new_obj_id] >= time_threshold):
                        class_1_count = 0
                        for det_inner in detections:
                            if int(det_inner.cls) == 1:  # Bottle detected
                                inner_bbox = det_inner.xyxy[0].tolist()
                                center_x_inner = (inner_bbox[0] + inner_bbox[2]) / 2
                                center_y_inner = (inner_bbox[1] + inner_bbox[3]) / 2
                                if bbox[0] <= center_x_inner <= bbox[2] and bbox[1] <= center_y_inner <= bbox[3]:
                                    class_1_count += 1

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

                elif class_id == 1:  # Bottle detected
                    # For bottles, if det.id is None, new_obj_id remains None
                    if det.id is not None:
                        if obj_id not in class_1_id_mapping:
                            class_1_id_mapping[obj_id] = class_1_id_counter
                            class_1_id_counter += 1
                        new_obj_id = class_1_id_mapping[obj_id]
                    detected_class_1.append(new_obj_id)

        # Remove boxes no longer detected from static tracking and previous bbox storage
        static_start_time = {k: v for k, v in static_start_time.items() if k in detected_class_0}
        class_0_prev_bbox = {k: v for k, v in class_0_prev_bbox.items() if k in detected_class_0}

        # Annotate the output frame with bounding boxes and labels (scale YOLO coordinates up)
        colors = {0: (0, 255, 0), 1: (0, 0, 255)}  # Green for boxes, red for bottles
        if results:
            detections = results[-1].boxes
            for det in detections:
                class_id = int(det.cls)
                # Get bounding box coordinates from YOLO frame and scale them
                x1, y1, x2, y2 = det.xyxy[0].tolist()
                x1 = int(x1 * scale_factor)
                y1 = int(y1 * scale_factor)
                x2 = int(x2 * scale_factor)
                y2 = int(y2 * scale_factor)
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

video_path=r"C:/Users/pp3tr/OneDrive/Desktop/bottle_demos/165534 MY HOME  YΠ.ΛΕΜΟΝΙ 425ML"

# Loop through each model in the list and process the video
for model_name in model_list:
    model_path = "C:/Users/pp3tr/OneDrive/Desktop/bottle_demos/best.pt" #f'/home/team/SSD2/Periklis/Projects/Bottles/Notebooks/YYolo_train/runs/detect/{model_name}/weights/best.pt/'
    print(f"Processing video with model: {model_name}")
    process_video(model_path, video_path)