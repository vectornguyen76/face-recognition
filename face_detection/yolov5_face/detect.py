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


class Detector(object):
    def __init__(self, model_path="face_detection/yolov5_face/yolov5n-0.5.pt"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = attempt_load(model_path, map_location=device)

        # Parameters
        self.size_convert = 128
        self.conf_thres = 0.4
        self.iou_thres = 0.5

    # Resize image
    def resize_image(self, img0, img_size):
        h0, w0 = img0.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size

        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=self.model.stride.max())  # check img_size
        img = letterbox(img0, new_shape=imgsz)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return img

    def scale_coords_landmarks(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(
                img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
            )  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                img1_shape[0] - img0_shape[0] * gain
            ) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
        coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
        coords[:, :10] /= gain
        # clip_coords(coords, img0_shape)
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

    def inference_detect(self, input_image):
        # Resize image
        img = self.resize_image(img0=input_image.copy(), img_size=self.size_convert)

        # Via yolov5-face
        with torch.no_grad():
            pred = self.model(img[None, :])[0]

        # Apply NMS
        det = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)[0]
        bboxs = np.int32(
            scale_coords(img.shape[1:], det[:, :4], input_image.shape)
            .round()
            .cpu()
            .numpy()
        )

        landmarks = np.int32(
            self.scale_coords_landmarks(img.shape[1:], det[:, 5:15], input_image.shape)
            .round()
            .cpu()
            .numpy()
        )

        return bboxs, landmarks

    def inference_tracking(self, input_image, timer):
        height, width = input_image.shape[:2]
        img_info = {"id": 0}
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = input_image

        # Resize image
        img = self.resize_image(img0=input_image.copy(), img_size=self.size_convert)

        # Via yolov5-face
        with torch.no_grad():
            timer.tic()
            outputs = self.model(img[None, :])[0]

        outputs = non_max_suppression_face(outputs, self.conf_thres, self.iou_thres)
        outputs[0] = outputs[0][:, :5]

        return outputs, img_info
