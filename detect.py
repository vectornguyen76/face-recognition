import time

import cv2

from face_detection.yolov5_face.detect import Detector


def main():
    detector = Detector()

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
        "results/face-detection.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, size
    )

    # Read until video is completed
    while True:
        # Capture frame-by-frame
        _, frame = cap.read()

        # Get faces
        bboxs, landmarks = detector.inference_detect(input_image=frame)
        h, w, c = frame.shape

        tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
        clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

        # Get boxs
        for i in range(len(bboxs)):
            # Get location face
            x1, y1, x2, y2 = bboxs[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)

            # Landmarks
            for x in range(5):
                point_x = int(landmarks[i][2 * x])
                point_y = int(landmarks[i][2 * x + 1])
                cv2.circle(frame, (point_x, point_y), tl + 1, clors[x], -1)

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

        # Save video
        video.write(frame)

        # Show result
        cv2.imshow("Face Detection", frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    video.release()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
