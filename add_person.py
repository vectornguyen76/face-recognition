import argparse
import os
import shutil

import cv2
import numpy as np
import torch
from torchvision import transforms

from face_alignment.utils import compare_encodings, norm_crop
from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet18

# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight = torch.load(
    "face_recognition/arcface/weights/arcface_r18.pth", map_location=device
)
model_emb = iresnet18()

model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()

# Initialize the face detector
detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5m-face.pt")


def read_features(feature_path):
    try:
        data = np.load(feature_path + ".npz", allow_pickle=True)
        images_name = data["images_name"]
        images_emb = data["images_emb"]

        return images_name, images_emb
    except:
        return None


def get_feature(face_image, training=True):
    face_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

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
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)
    return images_emb


def add_person(backup_dir, add_person_dir, faces_save_dir, features_path):
    # Init results output
    images_name = []
    images_emb = []

    # Read train folder, get and save face
    for name_person in os.listdir(add_person_dir):
        person_image_path = os.path.join(add_person_dir, name_person)

        # Create path save person face
        person_face_path = os.path.join(faces_save_dir, name_person)
        os.makedirs(person_face_path, exist_ok=True)

        for image_name in os.listdir(person_image_path):
            if image_name.endswith(("png", "jpg", "jpeg")):
                input_image = cv2.imread(os.path.join(person_image_path, image_name))

                # Get faces and landmarks using the face detector
                bboxes, landmarks = detector.detect(image=input_image)

                # Get boxs
                for i in range(len(bboxes)):
                    # Get number files in person path
                    number_files = len(os.listdir(person_face_path))

                    # Get location face
                    x1, y1, x2, y2, score = bboxes[i]

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

    features = read_features(features_path)

    if features != None:
        # Read features
        old_images_name, old_images_emb = features

        # Add feature and name of image to feature database
        images_name = np.hstack((old_images_name, images_name))
        images_emb = np.vstack((old_images_emb, images_emb))

        print("Update feature!")

    # Save features
    np.savez_compressed(features_path, images_name=images_name, images_emb=images_emb)

    # Move add person data to backup data
    for sub_dir in os.listdir(add_person_dir):
        dir_to_move = os.path.join(add_person_dir, sub_dir)
        shutil.move(dir_to_move, backup_dir, copy_function=shutil.copytree)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="./datasets/backup",
        help="dir folder save person data",
    )
    parser.add_argument(
        "--add-person-dir",
        type=str,
        default="./datasets/new_person",
        help="dir folder add new person",
    )
    parser.add_argument(
        "--faces-save-dir",
        type=str,
        default="./datasets/data/",
        help="dir folder save face",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="./datasets/face_features/feature",
        help="path save face features",
    )

    opt = parser.parse_args()
    return opt


def main(opt):
    add_person(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
