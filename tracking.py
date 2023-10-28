import os
import os.path as osp
import time

import cv2
import yaml
from loguru import logger

from face_detection.yolov5_face.detect import Detector
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.timer import Timer
from face_tracking.tracker.visualize import plot_tracking


def load_config(file_name):
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def inference(detector, args):
    cap = cv2.VideoCapture(args["input_path"])

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    save_folder = osp.join(args["output_path"], timestamp)

    os.makedirs(save_folder, exist_ok=True)

    if args["input_tupe"] == "video":
        save_path = osp.join(save_folder, args["input_path"].split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")

    logger.info(f"video save_path is {save_path}")

    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args=args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    while True:
        if frame_id % 20 == 0:
            logger.info(
                "Processing frame {} ({:.2f} fps)".format(
                    frame_id, 1.0 / max(1e-5, timer.average_time)
                )
            )
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = detector.inference_tracking(
                input_image=frame, timer=timer
            )

            if outputs[0] is not None:
                online_targets = tracker.update(
                    outputs[0], [img_info["height"], img_info["width"]], (128, 128)
                )
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args["aspect_ratio_thresh"]
                    if tlwh[2] * tlwh[3] > args["min_box_area"] and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                timer.toc()
                online_im = plot_tracking(
                    img_info["raw_img"],
                    online_tlwhs,
                    online_ids,
                    frame_id=frame_id + 1,
                    fps=1.0 / timer.average_time,
                )
            else:
                timer.toc()
                online_im = img_info["raw_img"]
            if True:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1


def main():
    file_name = "./face_tracking/config/config_tracking.yaml"
    config_tracking = load_config(file_name)
    detector = Detector()

    logger.info("Args: {}".format(config_tracking))
    inference(detector=detector, args=config_tracking)


if __name__ == "__main__":
    main()
