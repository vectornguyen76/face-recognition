import os
import sys

import cv2
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords


class Yolov5Face(object):
    def __init__(self, model_file=None):
        """
        Initialize the Detector class.

        :param model_path: Path to the YOLOv5 model file (default is yolov5n-0.5.pt)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = attempt_load(model_file, map_location=device)

        # Parameters
        self.size_convert = 128  # Size for image conversion
        self.conf_thres = 0.4  # Confidence threshold
        self.iou_thres = 0.5  # Intersection over Union threshold

    def resize_image(self, img0, img_size):
        """
        Resize the input image.

        :param img0: The input image to be resized.
        :param img_size: The desired size for the image.

        :return: The resized and preprocessed image.
        """
        h0, w0 = img0.shape[:2]  # Original height and width
        r = img_size / max(h0, w0)  # Resize image to img_size

        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=self.model.stride.max())  # Check img_size
        img = letterbox(img0, new_shape=imgsz)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return img

    def scale_coords_landmarks(self, img1_shape, coords, img0_shape, ratio_pad=None):
        """
        Rescale coordinates from img1_shape to img0_shape.

        :param img1_shape: Shape of the source image.
        :param coords: Coordinates to be rescaled.
        :param img0_shape: Shape of the target image.
        :param ratio_pad: Padding ratio.

        :return: Rescaled coordinates.
        """
        if ratio_pad is None:  # Calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                img1_shape[0] - img0_shape[0] * gain
            ) / 2
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
        coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
        coords[:, :10] /= gain
        coords[:, :10] = coords[:, :10].clamp(
            0, img0_shape[1]
        )  # Clamp x and y coordinates

        # Reshape the coordinates into the desired format
        coords = coords.reshape(-1, 5, 2)
        return coords

    def detect(self, image):
        """
        Perform face detection on the input image.

        :param input_image: The input image for face detection.

        :return: Detected bounding boxes and landmarks.
        """
        # Resize image
        img = self.resize_image(img0=image.copy(), img_size=self.size_convert)

        # Via yolov5-face
        with torch.no_grad():
            pred = self.model(img[None, :])[0]

        # Apply NMS
        det = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)[0]
        bboxes = np.int32(
            scale_coords(img.shape[1:], det[:, :5], image.shape).round().cpu().numpy()
        )

        landmarks = np.int32(
            self.scale_coords_landmarks(img.shape[1:], det[:, 5:15], image.shape)
            .round()
            .cpu()
            .numpy()
        )

        return bboxes, landmarks

    def detect_tracking(self, image):
        """
        Perform object tracking on the input image.

        :param input_image: The input image for object tracking.

        :return: Tracking results and image information.
        """
        height, width = image.shape[:2]
        img_info = {"id": 0}
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = image

        # Resize image
        img = self.resize_image(img0=image.copy(), img_size=self.size_convert)

        # Via yolov5-face
        with torch.no_grad():
            pred = self.model(img[None, :])[0]

        scale = min(
            img.shape[1] / float(image.shape[0]), img.shape[2] / float(image.shape[1])
        )

        # Apply NMS
        det = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)[0]

        bboxes = scale_coords(img.shape[1:], det[:, :4], image.shape)
        scores = det[:, 4:5]
        outputs = torch.cat((bboxes, scores), dim=1)
        outputs[:, :4] *= scale

        bboxes = np.int32(bboxes.round().cpu().numpy())

        landmarks = np.int32(
            self.scale_coords_landmarks(img.shape[1:], det[:, 5:15], image.shape)
            .round()
            .cpu()
            .numpy()
        )

        return outputs, img_info, bboxes, landmarks
