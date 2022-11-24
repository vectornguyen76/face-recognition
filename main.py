#pytorch
import torch
from torchvision import transforms

#other lib
import sys
import numpy as np
import os
import cv2
from insightface.insight_face import iresnet100
from PIL import Image

sys.path.insert(0, "yolov5_face")

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get model
# model = attempt_load("yolov5_face/yolov5m-face.pt", map_location=device)
model = attempt_load("yolov5_face/yolov5n-0.5.pt", map_location=device)

weight = torch.load("insightface/16_backbone.pth", map_location = device)
model_emb = iresnet100()
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

def training(input_image, name_persion):
    # input_image = cv2.imread(path_image) 

    # Path to save face
    path_database = "faces_database/"
    path_persion = os.path.join(path_database, name_persion)

    # Create dir
    os.makedirs(path_persion, exist_ok=True)

    images_name = []
    images_emb = []

    # Get faces
    bboxs = get_face(input_image)

    # Get boxs
    for i in range(len(bboxs)):
        # Get number files in persion path
        number_files = len(os.listdir(path_persion))

        # Get location face
        x1, y1, x2, y2 = bboxs[i]

        # Get face from location
        face_image = input_image[y1:y2, x1:x2]

        # Path save face
        path_save_face = path_persion + f"/{number_files}.jpg"
        
        # Save to face database 
        cv2.imwrite(path_save_face, face_image)
        
        # Get feature from face
        images_emb.append(get_feature(face_image, training=True))
        images_name.append(name_persion)
    
    # Convert to array
    images_emb = np.array(images_emb)
    images_name = np.array(images_name)
    
    # Read features
    old_images_name, old_images_emb = read_features()   
    
    # Add feature and name of image to feature database
    new_images_name = np.hstack((old_images_name, images_name))
    new_images_emb = np.vstack((old_images_emb, images_emb))
    
    # Save features
    path_features =  "static/feature/"
    np.savez_compressed(path_features + "face_features", 
                        arr1 = new_images_name, arr2 = new_images_emb)
    
def recognition(image):
    # Get faces
    bboxs = get_face(image)

    # Get boxs
    for i in range(len(bboxs)):
        # Get location face
        x1, y1, x2, y2 = bboxs[i]

        # Get face from location
        face_image = image[y1:y2, x1:x2]
        
        # Get feature from face
        query_emb = (get_feature(face_image, training=False))
        print(query_emb.shape)
        
        # Read features
        images_names, images_embs = read_features()   
        
        scores = (query_emb @ images_embs.T)[0]

        id_min = np.argmax(scores)
        score = scores[id_min]
        name = images_names[id_min]
        
        if score < 0.2:
            caption= "UN_KNOWN"
        else:
            caption = f"{name.split('_')[0].upper()}:{score:.2f}"

        t_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 146, 230), 3)
        cv2.rectangle(
            image, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1)
        cv2.putText(image, caption, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    
    return image

if __name__=="__main__":
    '''Traning'''
    # Input
    path_image = "train_image.jpg"
    name_persion = "Châu"
    image = cv2.imread(path_image)
    # Training
    training(image, name_persion)
    
    
    '''Recognition'''
    # Input
    path_query = "test_image.jpg"
    image = cv2.imread(path_query) 
    # Regnition
    result = recognition(image)
    cv2.imwrite("result.jpg", result)
    # Output
    cv2.imshow("Result", result)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image
    