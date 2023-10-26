import argparse
#pytorch
from concurrent.futures import thread
from xmlrpc.client import Boolean
import torch
from torchvision import transforms
from threading import Thread

#other lib
import sys
import numpy as np
import os
import cv2
import shutil

sys.path.insert(0, "face_detection/yolov5_face")
from face_detection.yolov5_face.models.experimental import attempt_load
from face_detection.yolov5_face.utils.datasets import letterbox
from face_detection.yolov5_face.utils.general import check_img_size, non_max_suppression_face, scale_coords

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get model detect
## Case 1:
# model = attempt_load("face_detection/yolov5_face/yolov5s-face.pt", map_location=device)

## Case 2:
model = attempt_load("face_detection/yolov5_face/yolov5n-0.5.pt", map_location=device)

# Get model recognition
## Case 1: 
from face_recognition.insightface.model import iresnet100
weight = torch.load("insightface/resnet100_backbone.pth", map_location = device)
model_emb = iresnet100()

## Case 2: 
# from face_recognition.insightface.model import iresnet18
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
    size_convert = 256
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

def read_features(root_fearure_path):
    try:
        data = np.load(root_fearure_path + ".npz", allow_pickle=True)
        images_name = data["arr1"]
        images_emb = data["arr2"]
        
        return images_name, images_emb
    except:
        return None

def training(full_training_dir, additional_training_dir, 
             faces_save_dir, features_save_dir, is_add_user):
    
    # Init results output
    images_name = []
    images_emb = []
    
    # Check mode full training or additidonal
    if is_add_user == True:
        source = additional_training_dir
    else:
        source = full_training_dir
    
    # Read train folder, get and save face 
    for name_person in os.listdir(source):
        person_image_path = os.path.join(source, name_person)
        
        # Create path save person face
        person_face_path = os.path.join(faces_save_dir, name_person)
        os.makedirs(person_face_path, exist_ok=True)
        
        for image_name in os.listdir(person_image_path):
            if image_name.endswith(("png", 'jpg', 'jpeg')):
                image_path = person_image_path + f"/{image_name}"
                input_image = cv2.imread(image_path)  # BGR 

                # Get faces
                bboxs = get_face(input_image)

                # Get boxs
                for i in range(len(bboxs)):
                    # Get number files in person path
                    number_files = len(os.listdir(person_face_path))

                    # Get location face
                    x1, y1, x2, y2 = bboxs[i]

                    # Get face from location
                    face_image = input_image[y1:y2, x1:x2]

                    # Path save face
                    path_save_face = person_face_path + f"/{number_files}.jpg"
                    
                    # Save to face database 
                    cv2.imwrite(path_save_face, face_image)
                    
                    # Get feature from face
                    images_emb.append(get_feature(face_image, training=True))
                    images_name.append(name_person)
    
    # Convert to array
    images_emb = np.array(images_emb)
    images_name = np.array(images_name)
    
    features = read_features(features_save_dir) 
    if features == None or is_add_user== False:
        pass
    else:        
        # Read features
        old_images_name, old_images_emb = features  
    
        # Add feature and name of image to feature database
        images_name = np.hstack((old_images_name, images_name))
        images_emb = np.vstack((old_images_emb, images_emb))
        
        print("Update feature!")
    
    # Save features
    np.savez_compressed(features_save_dir, 
                        arr1 = images_name, arr2 = images_emb)
    
    # Move additional data to full train data
    if is_add_user == True:
        for sub_dir in os.listdir(additional_training_dir):
            dir_to_move = os.path.join(additional_training_dir, sub_dir)
            shutil.move(dir_to_move, full_training_dir, copy_function = shutil.copytree)
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-training-dir', type=str, default='./database/full-training-datasets/', help='dir folder full training')
    parser.add_argument('--additional-training-dir', type=str, default='./database/additional-training-datasets/', help='dir folder additional training')
    parser.add_argument('--faces-save-dir', type=str, default='./database/face-datasets/', help='dir folder save face features')
    parser.add_argument('--features-save-dir', type=str, default='./static/feature/face_features', help='dir folder save face features')
    parser.add_argument('--is-add-user', type=bool, default=True, help='Mode add user or full training')
    
    opt = parser.parse_args()
    return opt

def main(opt):
    training(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

    