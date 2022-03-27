#pytorch
import torch
from torchvision import transforms

#other lib
import sys
import numpy as np
import os
import cv2

sys.path.insert(0, "yolov5_face")

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get model
model = attempt_load("yolov5_face/yolov5m-face.pt", map_location=device)

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

if __name__=="__main__":
    # Path read image from datasets
    root_path = "datasets"

    # Path save face database
    save_path = "faces_database"
    os.makedirs(save_path, exist_ok=True)

    # Read and save face 
    for folder in os.listdir(root_path):
        if os.path.isdir(root_path + "/"+ folder):
            # Get path persion face 
            persion_path = f"{save_path}/{folder}"
            os.makedirs(persion_path, exist_ok=True)
            
            for name in os.listdir(root_path + "/" + folder):
                if name.endswith(("png", 'jpg', 'jpeg')):
                    path = f"{root_path}/{folder}/{name}"
                    orgimg = cv2.imread(path)  # BGR 

                    # Resize image
                    img = resize_image(orgimg.copy(), size_convert)

                    # Via yolov5-face
                    with torch.no_grad():
                        pred = model(img[None, :])[0]
                        
                    # Apply NMS
                    det = non_max_suppression_face(pred, conf_thres, iou_thres)[0]
                    bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], orgimg.shape).round().cpu().numpy())

                    # Get boxs
                    for i in range(len(bboxs)):
                        number_files = len(os.listdir(persion_path))

                        x1, y1, x2, y2 = bboxs[i]

                        image = orgimg[y1:y2, x1:x2]

                        name = persion_path + f"/{number_files}.jpg"
                                
                        cv2.imwrite(name, image)