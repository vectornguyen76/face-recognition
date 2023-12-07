import os
import os.path as osp
import time
import requests
import numpy as np
import cv2
import imutils
import yaml
from face_detection.yolov5_face.detector import Yolov5Face
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.timer import Timer
from face_tracking.tracker.visualize import plot_tracking
import redis
from loguru import logger
import json
import threading
from face_alignment.utils import norm_crop, compare_encodings
import torch
from torchvision import transforms
from face_detection.scrfd.detector import SCRFD
from queue import Queue
from face_recognition.arcface.model import iresnet18


face_data = {}
detector = Yolov5Face(model_file="face_detection/yolov5_face/yolov5n-face.pt")
# detector = SCRFD(model_file="face_detection/scrfd/scrfd_2.5g_bnkps.onnx")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight = torch.load(
    "face_recognition/arcface/resnet18_backbone.pth", map_location=device
)
model_emb = iresnet18()

model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()


# Function to load a YAML configuration file
def load_config(file_name):
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def track(frame, detector, tracker, timer, args, frame_id):
    # Perform face detection and tracking on the frame
    outputs, img_info, bboxs, landmarks = detector.detect_tracking(
        image=frame, timer=timer
    )

    online_tlwhs = []
    online_ids = []
    online_scores = []

    if outputs[0] is not None:
        online_targets = tracker.update(
            outputs[0], [img_info["height"], img_info["width"]], (128, 128)
        )

        for i in range(len(online_targets)):
            t = online_targets[i]
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > args["aspect_ratio_thresh"]
            if tlwh[2] * tlwh[3] > args["min_box_area"] and not vertical:
                # get tlwh from bboxs
                tlwh = np.asarray(
                    [
                        bboxs[i][0],
                        bboxs[i][1],
                        bboxs[i][2] - bboxs[i][0],
                        bboxs[i][3] - bboxs[i][1],
                    ]
                )
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)

        timer.toc()

        online_im = plot_tracking(
            img_info["raw_img"],
            online_tlwhs,
            online_ids,
            names=face_data,
            frame_id=frame_id + 1,
            fps=1.0 / timer.average_time,
        )
    else:
        timer.toc()
        online_im = img_info["raw_img"]

    return online_im, online_tlwhs, online_ids, img_info["raw_img"], bboxs, landmarks


face_preprocess = transforms.Compose(
    [
        transforms.ToTensor(),  # input PIL => (3,56,56), /255.0
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


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


def get_face(input_image):
    bboxs, landmarks = detector.detect(image=input_image)
    return bboxs, landmarks


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


def recognition(face_image):
    # Get feature from face
    query_emb = get_feature(face_image, training=False)

    # scores = (query_emb @ images_embs.T)[0]
    score, id_min = compare_encodings(query_emb, images_embs)

    name = images_names[id_min]
    score = score[0][0]
    return score, name


images_names, images_embs = read_features()


def IPCamera(url = "http://192.168.226.76:8080/shot.jpg"):
    # url = "http://192.168.1.9:8080/shot.jpg"
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    # img = imutils.resize(img, width=1000, height=1800)
    return img

def Camera(cap = cv2.VideoCapture(1)):
    # Capture frame-by-frame
    ret, frame = cap.read()
    return frame

# thread 1
def inference(detector, args, frame_queue):
    # Initialize a tracker and a timer
    tracker = BYTETracker(args=args, frame_rate=30)
    timer = Timer()
    frame_id = 0

    while True:
        img = IPCamera()
        # img = Camera()

        online_frame, online_tlwhs, online_ids, raw_image, bboxs, landmarks = track(
            img, detector, tracker, timer, args, frame_id
        )
        frame_queue.put((raw_image, online_ids, bboxs, landmarks))
        cv2.imshow("Android_cam", online_frame)

        # Press Esc key to exit
        if cv2.waitKey(1) == 27:
            break


# thread 2
def recognize(frame_queue):
    while True:
        raw_image, online_ids, bboxs, landmarks = frame_queue.get()
        for i in range(len(bboxs)):
            # Get Face Alignment from location
            align = norm_crop(raw_image, landmarks[i])

            score, name = recognition(align)

            if name != None:
                if score < 0.25:
                    caption = "UN_KNOWN"
                else:
                    caption = f"{name}:{score:.2f}"
            face_data[online_ids[i]] = caption
        print(face_data)


# Main function
def main():
    file_name = "./face_tracking/config/config_tracking.yaml"
    config_tracking = load_config(file_name)

    frame_queue = Queue()
    # inference(detector=detector, args=config_tracking)
    thread_track = threading.Thread(
        target=inference,
        args=(
            detector,
            config_tracking,
            frame_queue,
        ),
    )
    thread_track.start()

    thread_recognize = threading.Thread(target=recognize, args=(frame_queue,))
    thread_recognize.start()


if __name__ == "__main__":
    # conn = redis.Redis(host="localhost", port=6379)
    # if not conn.ping():
    #     raise Exception("Redis unavailable")
    # print("[x] Redis connected")
    # conn.delete("track_face")
    main()
