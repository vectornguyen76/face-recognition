import argparse
import yaml
import os
import os.path as osp
import time
import sys
import cv2
import torch
from loguru import logger
sys.path.insert(0, "face_tracking")
from face_tracking.tracker.visualize import plot_tracking
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.timer import Timer

sys.path.insert(0, "face_detection/yolov5_face")
from face_detection.yolov5_face.models.experimental import attempt_load
from face_detection.yolov5_face.utils.datasets import letterbox
from face_detection.yolov5_face.utils.general import check_img_size, non_max_suppression_face, scale_coords

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get model detect
## Case 1:
# model = attempt_load("face_detection/yolov5_face/yolov5s-face.pt", map_location=device)

## Case 2:
model = attempt_load("face_detection/yolov5_face/yolov5n-0.5.pt", map_location=device)

def load_config(file_name):
    with open(file_name, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

# Resize image
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

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
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


def get_face(input_image, timer):
    height, width = input_image.shape[:2]
    img_info = {"id": 0}
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = input_image
    
    # Parameters
    size_convert = 128
    conf_thres = 0.4
    iou_thres = 0.5
    
    # Resize image
    img = resize_image(input_image.copy(), size_convert)
    
    # Via yolov5-face
    with torch.no_grad():
        timer.tic()
        outputs = model(img[None, :])[0]

    # Apply NMS
    outputs = non_max_suppression_face(outputs, conf_thres, iou_thres)
    outputs[0] = outputs[0][:,:5]
    
    return outputs, img_info


def inference(args):
    cap = cv2.VideoCapture(args['input_path'])
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    save_folder = osp.join(args['output_path'], timestamp)
    
    os.makedirs(save_folder, exist_ok=True)
    
    if args['input_tupe'] == "video":
        save_path = osp.join(save_folder, args['input_path'].split("/")[-1])
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
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = get_face(input_image=frame, timer=timer)         
            
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], (128,128))
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args['aspect_ratio_thresh']
                    if tlwh[2] * tlwh[3] > args['min_box_area'] and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if True:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1


def main():
    file_name = './face_tracking/config/config_tracking.yaml'
    config_tracking = load_config(file_name)

    logger.info("Args: {}".format(config_tracking))
    
    inference(args=config_tracking)


if __name__ == "__main__":
    main()