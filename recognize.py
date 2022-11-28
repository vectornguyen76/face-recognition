#pytorch
from concurrent.futures import thread
from sqlalchemy import null
import torch
from torchvision import transforms
import time
from threading import Thread

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

# Get model detect
## Case 1:
# model = attempt_load("yolov5_face/yolov5s-face.pt", map_location=device)

## Case 2:
model = attempt_load("yolov5_face/yolov5n-0.5.pt", map_location=device)

# Get model recognition
## Case 1: 
from insightface.insight_face import iresnet100
weight = torch.load("insightface/resnet100_backbone.pth", map_location = device)
model_emb = iresnet100()

## Case 2: 
# from insightface.insight_face import iresnet18
# weight = torch.load("insightface/resnet18_backbone.pth", map_location = device)
# model_emb = iresnet18()

model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()

face_preprocess = transforms.Compose([
                                    transforms.ToTensor(), # input PIL => (3,56,56), /255.0
                                    transforms.Resize((112, 112)),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])

isThread = True
score = 0
name = null

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

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def get_face(input_image):
    # Parameters
    size_convert = 128
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
    
    landmarks = np.int32(scale_coords_landmarks(img.shape[1:], det[:, 5:15], input_image.shape).round().cpu().numpy())    
    
    return bboxs, landmarks

def get_feature(face_image, training = True): 
    # Convert to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Preprocessing image BGR
    face_image = face_preprocess(face_image).to(device)
    
    # Via model to get feature
    with torch.no_grad():
        if training:
            emb_img_face = model_emb(face_image[None, :])[0].cpu().numpy()
        else:
            emb_img_face = model_emb(face_image[None, :]).cpu().numpy()
    
    # Convert to array
    images_emb = emb_img_face/np.linalg.norm(emb_img_face)
    return images_emb

def read_features(root_fearure_path = "static/feature/face_features.npz"):
    data = np.load(root_fearure_path, allow_pickle=True)
    images_name = data["arr1"]
    images_emb = data["arr2"]
    
    return images_name, images_emb

def recognition(face_image):
    global isThread, score, name
    
    # Get feature from face
    query_emb = (get_feature(face_image, training=False))
    
    # Read features
    images_names, images_embs = read_features()   

    scores = (query_emb @ images_embs.T)[0]

    id_min = np.argmax(scores)
    score = scores[id_min]
    name = images_names[id_min]
    isThread = True
    print("successful")



def main():
    global isThread, score, name
    
    # Open camera 
    cap = cv2.VideoCapture(0)
    start = time.time_ns()
    frame_count = 0
    fps = -1
    
    # Save video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    size = (frame_width, frame_height)
    video = cv2.VideoWriter('./static/results/face-recognition2.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 6, size)
    
    # Read until video is completed
    while(True):
        # Capture frame-by-frame
        _, frame = cap.read()
        
        # Get faces
        bboxs, landmarks = get_face(frame)
        h, w, c = frame.shape
        
        tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
        clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
        
        # Get boxs
        for i in range(len(bboxs)):
            # Get location face
            x1, y1, x2, y2 = bboxs[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
            
            # Landmarks
            for x in range(5):
                point_x = int(landmarks[i][2 * x])
                point_y = int(landmarks[i][2 * x + 1])
                cv2.circle(frame, (point_x, point_y), tl+1, clors[x], -1)
            
            # Get face from location
            if isThread == True:
                isThread = False
                
                # Recognition
                face_image = frame[y1:y2, x1:x2]
                thread = Thread(target=recognition, args=(face_image,))
                thread.start()
                
                thread.join()
        
            if name == null:
                continue
            else:
                if score < 0.25:
                    caption= "UN_KNOWN"
                else:
                    caption = f"{name.split('_')[0].upper()}:{score:.2f}"

                t_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                
                cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1)
                cv2.putText(frame, caption, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)       
            
        
        # Count fps 
        frame_count += 1
        
        if frame_count >= 30:
            end = time.time_ns()
            fps = 1e9 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()
    
        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        
        video.write(frame)
        cv2.imshow("Face Recognition", frame)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  
    
    video.release()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)

if __name__=="__main__":
    main()