# pytorch
# other lib
import os
import sys
import time

import cv2
import numpy as np
import torch
from torchvision import transforms

from face_alignment.utils import compare_encodings, norm_crop
from face_detection.scrfd.detector import SCRFD

# sys.path.insert(0, "face_detection/yolov5_face")

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get model detection
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

# Get model recognition
from face_recognition.arcface.model import iresnet18

weight = torch.load(
    "face_recognition/arcface/weights/arcface_r18.pth", map_location=device
)
model_emb = iresnet18()

model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()

face_preprocess = transforms.Compose(
    [
        transforms.ToTensor(),  # input PIL => (3,56,56), /255.0
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def get_face(input_image):
    bboxs, landmarks = detector.detect(image=input_image)
    return bboxs, landmarks


def get_feature(face_image, training=True):
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


# def read_features(root_fearure_path="static/feature/face_features.npz"):
#     # data = np.load(root_fearure_path, allow_pickle=True)
#     images_name = data["arr1"]
#     images_emb = data["arr2"]
#     return images_name, images_emb


# test
def read_features(root_fearure_path="database/face-datasets"):
    # data = np.load(root_fearure_path, allow_pickle=True)
    images_emb = []
    images_name = []
    for name in os.listdir(root_fearure_path):
        print("name", name)
        for img in os.listdir(os.path.join(root_fearure_path, name)):
            image = cv2.imread(os.path.join(root_fearure_path, name, img))
            bboxs, landmarks = get_face(image)
            align = norm_crop(image, landmarks[0])
            embed = get_feature(align, training=False)
            images_emb.append(embed)
            images_name.append(name)
    return np.array(images_name), np.array(images_emb)


images_names, images_embs = read_features()


def recognition(face_image):
    # Get feature from face
    query_emb = get_feature(face_image, training=False)

    # scores = (query_emb @ images_embs.T)[0]
    score, id_min = compare_encodings(query_emb, images_embs)

    name = images_names[id_min]
    score = score[0][0]
    return score, name


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
    video = cv2.VideoWriter(
        "./static/results/face-recognition2.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        6,
        size,
    )

    # Read until video is completed
    while True:
        # Capture frame-by-frame
        _, frame = cap.read()

        # Get faces
        bboxs, landmarks = get_face(frame)
        h, w, c = frame.shape

        tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
        clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

        # Get boxs
        for i in range(len(bboxs)):
            # Get location face
            x1, y1, x2, y2, _ = bboxs[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)

            # Landmarks
            for id, key_point in enumerate(landmarks[i]):
                cv2.circle(frame, tuple(key_point), tl + 1, clors[id], -1)

            # Get Face Alignment from location
            align = norm_crop(frame, landmarks[i])

            score, name = recognition(align)

            if name == None:
                continue
            else:
                if score < 0.25:
                    caption = "UN_KNOWN"
                else:
                    caption = f"{name}:{score:.2f}"

                t_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]

                cv2.rectangle(
                    frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1
                )
                cv2.putText(
                    frame,
                    caption,
                    (x1, y1 + t_size[1]),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    [255, 255, 255],
                    2,
                )

        # Count fps
        frame_count += 1

        if frame_count >= 30:
            end = time.time_ns()
            fps = 1e9 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()

        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(
                frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        video.write(frame)
        cv2.imshow("Face Recognition", frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    video.release()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
