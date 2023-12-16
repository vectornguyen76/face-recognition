import threading
import time

import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms

from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.visualize import plot_tracking

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)

images_names, images_embs = read_features(feature_path="./datasets/face_features/feature")

id_face_mapping = {}

data_mapping = {
    "raw_image": [],
    "tracking_ids": [],
    "detection_bboxes": [],
    "detection_landmarks": [],
    "tracking_bboxes": [],
}


# Function to load a YAML configuration file
def load_config(file_name):
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def process_tracking(frame, detector, tracker, args, frame_id, fps):
    # Perform face detection and tracking on the frame
    outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)

    tracking_tlwhs = []
    tracking_ids = []
    tracking_scores = []
    tracking_bboxes = []

    if outputs is not None:
        online_targets = tracker.update(
            outputs, [img_info["height"], img_info["width"]], (128, 128)
        )

        for i in range(len(online_targets)):
            t = online_targets[i]
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > args["aspect_ratio_thresh"]
            if tlwh[2] * tlwh[3] > args["min_box_area"] and not vertical:
                x1, y1, w, h = tlwh
                tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
                tracking_tlwhs.append(tlwh)
                tracking_ids.append(tid)
                tracking_scores.append(t.score)

        tracking_image = plot_tracking(
            img_info["raw_img"],
            tracking_tlwhs,
            tracking_ids,
            names=id_face_mapping,
            frame_id=frame_id + 1,
            fps=fps,
        )
    else:
        tracking_image = img_info["raw_img"]

    data_mapping["raw_image"] = img_info["raw_img"]
    data_mapping["detection_bboxes"] = bboxes
    data_mapping["detection_landmarks"] = landmarks
    data_mapping["tracking_ids"] = tracking_ids
    data_mapping["tracking_bboxes"] = tracking_bboxes

    return tracking_image


def get_feature(face_image):
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
        emb_img_face = recognizer(face_image[None, :]).cpu().numpy()

    # Convert to array
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)

    return images_emb


def recognition(face_image):
    # Get feature from face
    query_emb = get_feature(face_image)

    score, id_min = compare_encodings(query_emb, images_embs)
    name = images_names[id_min]
    score = score[0]

    return score, name


def mapping_bbox(box1, box2):
    # box format: (x_min, y_min, x_max, y_max)

    # Calculate the intersection area
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(
        0, y_max_inter - y_min_inter + 1
    )

    # Calculate the area of each bounding box
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the union area
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


# thread 1
def tracking(detector, args):
    # Initialize variables for measuring frame rate
    start_time = time.time_ns()
    frame_count = 0
    fps = -1

    # Initialize a tracker and a timer
    tracker = BYTETracker(args=args, frame_rate=30)
    frame_id = 0

    cap = cv2.VideoCapture(0)

    while True:
        _, img = cap.read()

        tracking_image = process_tracking(img, detector, tracker, args, frame_id, fps)

        # Calculate and display the frame rate
        frame_count += 1
        if frame_count >= 30:
            fps = 1e9 * frame_count / (time.time_ns() - start_time)
            frame_count = 0
            start_time = time.time_ns()

        cv2.imshow("Face Recognition", tracking_image)

        # Check for user exit input
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


# thread 2
def recognize():
    while True:
        raw_image = data_mapping["raw_image"]
        detection_landmarks = data_mapping["detection_landmarks"]
        detection_bboxes = data_mapping["detection_bboxes"]
        tracking_ids = data_mapping["tracking_ids"]
        tracking_bboxes = data_mapping["tracking_bboxes"]

        for i in range(len(tracking_bboxes)):
            for j in range(len(detection_bboxes)):
                mapping_score = mapping_bbox(box1=tracking_bboxes[i], box2=detection_bboxes[j])
                if mapping_score > 0.9:
                    face_alignment = norm_crop(img=raw_image, landmark=detection_landmarks[j])

                    score, name = recognition(face_image=face_alignment)
                    if name != None:
                        if score < 0.25:
                            caption = "UN_KNOWN"
                        else:
                            caption = f"{name}:{score:.2f}"

                    id_face_mapping[tracking_ids[i]] = caption

                    detection_bboxes = np.delete(detection_bboxes, j, axis=0)
                    detection_landmarks = np.delete(detection_landmarks, j, axis=0)

                    break

        if tracking_bboxes == []:
            print("Waiting person...")


def main():
    file_name = "./face_tracking/config/config_tracking.yaml"
    config_tracking = load_config(file_name)

    thread_track = threading.Thread(
        target=tracking,
        args=(
            detector,
            config_tracking,
        ),
    )
    thread_track.start()

    thread_recognize = threading.Thread(target=recognize)
    thread_recognize.start()


if __name__ == "__main__":
    main()
