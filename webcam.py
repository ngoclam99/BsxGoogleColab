from PIL import Image
import cv2
import torch
import math
import function.utils_rotate as utils_rotate
from IPython.display import display
import os
import time
import argparse
import function.helper as helper

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument('--save-video', action='store_true', help='Save output video instead of showing')
ap.add_argument('--camera-id', type=int, default=0, help='Camera ID (default: 0)')
ap.add_argument('--output', type=str, default='result/webcam_output.mp4', help='Output video path')
args = ap.parse_args()

# load model
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

prev_frame_time = 0
new_frame_time = 0

vid = cv2.VideoCapture(args.camera_id)

# Setup video writer if saving video
video_writer = None
if args.save_video:
    # Get video properties
    fps = int(vid.get(cv2.CAP_PROP_FPS)) or 30
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    print(f"Saving video to: {args.output}")
    print("Press 'q' to stop recording...")
else:
    print("GUI mode disabled due to OpenCV limitations.")
    print("Use --save-video flag to save output video instead.")
    print("Press Ctrl+C to stop...")
# vid = cv2.VideoCapture("1.mp4")
while(True):
    ret, frame = vid.read()

    plates = yolo_LP_detect(frame, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()
    for plate in list_plates:
        flag = 0
        x = int(plate[0]) # xmin
        y = int(plate[1]) # ymin
        w = int(plate[2] - plate[0]) # xmax - xmin
        h = int(plate[3] - plate[1]) # ymax - ymin
        crop_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
        cv2.imwrite("crop.jpg", crop_img)
        rc_image = cv2.imread("crop.jpg")
        lp = ""
        for cc in range(0,2):
            for ct in range(0,2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.add(lp)
                    cv2.putText(frame, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    flag = 1
                    break
            if flag == 1:
                break
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # Print detected plates to console
    if list_read_plates:
        print(f"Detected plates: {list(list_read_plates)}")

    if args.save_video and video_writer is not None:
        video_writer.write(frame)
        # Check for 'q' key press to stop recording
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    else:
        # Save frame periodically when not in video mode
        if int(time.time()) % 5 == 0:  # Save every 5 seconds
            timestamp = int(time.time())
            cv2.imwrite(f'result/frame_{timestamp}.jpg', frame)

        # Use a simple delay instead of waitKey for GUI
        time.sleep(0.1)

        # You can add a break condition here if needed
        # For now, use Ctrl+C to stop

# Cleanup
vid.release()
if video_writer is not None:
    video_writer.release()
    print(f"Video saved successfully to: {args.output}")
print("Webcam detection stopped.")
