from networks import FaceBox
from encoderl import DataEncoder
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
print('opencv version', cv2.__version__)


use_gpu = True
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX


def detect(im):
    im = cv2.resize(im, (1024,1024))
    im_tensor = torch.from_numpy(im.transpose((2,0,1)))
    im_tensor = im_tensor.float().div(255)
    # print(im_tensor.shape)
    if use_gpu:
        loc, conf = net(Variable(torch.unsqueeze(im_tensor, 0), volatile=True).cuda())
        loc, conf = loc.cpu(), conf.cpu()
    else:
        loc, conf = net(Variable(torch.unsqueeze(im_tensor, 0), volatile=True))

    boxes, labels, probs = data_encoder.decode(loc.data.squeeze(0),
                                                F.softmax(conf.squeeze(0)).data)
    return boxes, probs

if __name__ == "__main__":
    net = FaceBox()
    net.load_state_dict(torch.load('ckpt/faceboxes.pt', map_location=lambda storage, loc:storage))
    
    if use_gpu:
        net.cuda()
    net.eval()
    data_encoder = DataEncoder()
    # Create a VideoCapture object to access the camera (0 is usually the built-in webcam, but it can be a different number for external cameras)
    cap = cv2.VideoCapture(1)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read a frame.")
            break

        h,w,_ = frame.shape
        boxes, probs = detect(frame)
        print(boxes)
        for i, (box) in enumerate(boxes):
            print('i', i, 'box', box)
            x1 = int(box[0]*w)
            x2 = int(box[2]*w)
            y1 = int(box[1]*h)
            y2 = int(box[3]*h)
            print(x1, y1, x2, y2, w, h)
            cv2.rectangle(frame,(x1,y1+4),(x2,y2),(0,0,255),2)
            cv2.putText(frame, str(np.round(probs[i],2)), (x1,y1), font, 0.4, (0,0,255))

        cv2.imshow('Camera Feed', frame)

        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close the window
    cap.release()
    cv2.destroyAllWindows()