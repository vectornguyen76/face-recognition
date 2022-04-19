#pytorch
import torch
from torchvision import transforms

#other lib
import sys
import numpy as np
import os
import cv2
import time

sys.path.insert(0, "yolov5_face")

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get model
model = attempt_load("yolov5_face/yolov5n-0.5.pt", map_location=device)

# Parameter for yolov5
size_convert = 640  # setup size de day qua model
conf_thres = 0.4
iou_thres = 0.5

# Resize image
def resize_image(img0, img_size):
    h0, w0 = img0.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    return img

def get_face(input_image):
    # Parameters
    size_convert = 640 
    conf_thres = 0.4
    iou_thres = 0.5
    
    # Resize image
    img = resize_image(input_image.copy(), size_convert)

    # Via yolov5-face
    with torch.no_grad():
        pred = model(img[None, :])[0]

    # Apply NMS
    det = non_max_suppression_face(pred, conf_thres, iou_thres)[0]
    bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], input_image.shape).round().cpu().numpy())
    
    return bboxs

def main():
    # Open camera 
    cap = cv2.VideoCapture(0)
    start = time.time_ns()
    frame_count = 0
    fps = -1
    
    # Save video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    size = (frame_width, frame_height)
    video = cv2.VideoWriter('results/face-detection.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 6, size)
    
    # Read until video is completed
    while(True):
        # Capture frame-by-frame
        _, frame = cap.read()
        # Get faces
        bboxs = get_face(frame)

        # Get boxs
        for i in range(len(bboxs)):
            # Get location face
            x1, y1, x2, y2 = bboxs[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
            frame_count += 1
            
            # Count fps 
            if frame_count >= 30:
                end = time.time_ns()
                fps = 1e9 * frame_count / (end - start)
                frame_count = 0
                start = time.time_ns()
        
            if fps > 0:
                fps_label = "FPS: %.2f" % fps
                cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        video.write(frame)
        cv2.imshow("Face Detection", frame)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  
    
    video.release()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)

if __name__=="__main__":
    main()
