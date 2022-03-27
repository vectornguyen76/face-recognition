import torch
from torchvision import transforms
import os

from insightface.insight_face import iresnet100
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight = torch.load("insightface/16_backbone.pth", map_location = device)
model_emb = iresnet100()
model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()

# Preprocess Face
face_preprocess = transforms.Compose([
                                    transforms.ToTensor(), # input PIL => (3,56,56), /255.0
                                    transforms.Resize((112, 112)),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])

# Get database vectors
def inference_database(root_path = "faces_database"):
    
    images_name = []
    images_emb = []
    
    for folder in os.listdir(root_path):
        for name in os.listdir(root_path + "/" + folder):
            # Get path
            path = f"{root_path}/{folder}/{name}"
            
            # Preprocessing image
            img_face = face_preprocess(Image.open(path).convert("RGB")).to(device)
            
            # Via model to get feature
            with torch.no_grad():
                emb_img_face = model_emb(img_face[None, :])[0].cpu().numpy()
            
            # Add to list
            images_emb.append(emb_img_face)
            images_name.append(folder)
    
    # Convert to array
    images_emb = np.array(images_emb)
    images_name = np.array(images_name)
    
    return images_name, images_emb/np.linalg.norm(images_emb, axis=1, keepdims=True)

if __name__=="__main__":
    # Get names and vectors
    images_name, images_emb = inference_database()
    
    # Save features
    path_features =  "static/feature/"
    os.makedirs(path_features, exist_ok=True)
    np.savez_compressed(path_features + "face_features", 
                        arr1 = images_name, arr2 = images_emb)