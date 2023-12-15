import time

import cv2

from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face

# Initialize the face detector
detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5m-face.pt")
# detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")


def main():
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Initialize variables for measuring frame rate
    start = time.time_ns()
    frame_count = 0
    fps = -1

    # Save video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    video = cv2.VideoWriter(
        "results/face-detection.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, size
    )

    # Read frames from the camera
    while True:
        # Capture a frame from the camera
        _, frame = cap.read()

        # Get faces and landmarks using the face detector
        bboxes, landmarks = detector.detect(image=frame)
        h, w, c = frame.shape

        tl = 1 or round(0.002 * (h + w) / 2) + 1  # Line and font thickness
        clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

        # Draw bounding boxes and landmarks on the frame
        for i in range(len(bboxes)):
            # Get location of the face
            x1, y1, x2, y2, score = bboxes[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)

            # Draw facial landmarks
            for id, key_point in enumerate(landmarks[i]):
                cv2.circle(frame, tuple(key_point), tl + 1, clors[id], -1)

        # Calculate and display the frame rate
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

        # Save the frame to the video
        video.write(frame)

        # Show the result in a window
        cv2.imshow("Face Detection", frame)

        # Press 'Q' on the keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    # Release video and camera, and close all OpenCV windows
    video.release()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
