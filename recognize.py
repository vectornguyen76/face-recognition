import threading
import time
from queue import Queue

import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms

from add_person import read_features
from face_alignment.utils import compare_encodings, norm_crop
from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet18
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.visualize import plot_tracking

face_data = {}
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

images_names, images_embs = read_features(
    feature_path="./datasets/face_features/feature"
)


# Function to load a YAML configuration file
def load_config(file_name):
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def track(frame, detector, tracker, args, frame_id, fps):
    # Perform face detection and tracking on the frame
    outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)

    online_tlwhs = []
    online_ids = []
    online_scores = []
    online_bboxes = []

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
                online_bboxes.append([x1, y1, x1 + w, y1 + h])
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)

        online_im = plot_tracking(
            img_info["raw_img"],
            online_tlwhs,
            online_ids,
            names=face_data,
            frame_id=frame_id + 1,
            fps=fps,
        )
    else:
        online_im = img_info["raw_img"]

    return (
        online_bboxes,
        online_im,
        online_tlwhs,
        online_ids,
        img_info["raw_img"],
        bboxes,
        landmarks,
    )


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
        emb_img_face = model_emb(face_image[None, :]).cpu().numpy()

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
def inference(detector, args, frame_queue):
    # Initialize variables for measuring frame rate
    start_time = time.time_ns()
    frame_count = 0
    fps = -1

    # Initialize a tracker and a timer
    tracker = BYTETracker(args=args, frame_rate=30)
    frame_id = 0

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()

        (
            online_bboxes,
            online_frame,
            online_tlwhs,
            online_ids,
            raw_image,
            bboxes,
            landmarks,
        ) = track(img, detector, tracker, args, frame_id, fps)
        frame_queue.put((raw_image, online_ids, bboxes, landmarks, online_bboxes))

        # Calculate and display the frame rate
        frame_count += 1
        if frame_count >= 30:
            fps = 1e9 * frame_count / (time.time_ns() - start_time)
            frame_count = 0
            start_time = time.time_ns()

        cv2.imshow("Face Recognition", online_frame)

        # Check for user exit input
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


# thread 2
def recognize(frame_queue):
    while True:
        raw_image, online_ids, bboxes, landmarks, online_bboxes = frame_queue.get()

        for i in range(len(online_bboxes)):
            for j in range(len(bboxes)):
                mapping_score = mapping_bbox(online_bboxes[i], bboxes[j])
                if mapping_score > 0.9:
                    align = norm_crop(raw_image, landmarks[j])

                    score, name = recognition(align)

                    if name != None:
                        if score < 0.25:
                            caption = "UN_KNOWN"
                        else:
                            caption = f"{name}:{score:.2f}"
                    face_data[online_ids[i]] = caption

                    bboxes = np.delete(bboxes, j, axis=0)
                    landmarks = np.delete(landmarks, j, axis=0)
                    break


def main():
    file_name = "./face_tracking/config/config_tracking.yaml"
    config_tracking = load_config(file_name)

    frame_queue = Queue()

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
    main()
