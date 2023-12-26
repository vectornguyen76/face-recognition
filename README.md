# Real-Time Face Recognition

   <p align="center">
   <img src="./assets/face-recognition.gif" alt="Face Recognition" />
   <br>
   <em>Face Recognition</em>
   </p>

## Table of Contents

- [Architecture](#architecture)
- [How to use](#how-to-use)
  - [Create Environment and Install Packages](#create-environment-and-install-packages)
  - [Add new persons to datasets](#add-new-persons-to-datasets)
- [Technology](#technology)
  - [Face Detection](#face-detection)
  - [Face Recognition](#face-recognition)
  - [Face Tracking](#face-tracking)
  - [Matching Algorithm](#matching-algorithm)
- [Reference](#reference)

## Architecture

   <p align="center">
   <img src="./assets/sequence-diagram.png" alt="Sequence Diagram" />
   <br>
   <em>Sequence Diagram</em>
   </p>

## How to use

### Create Environment and Install Packages

```shell
conda create -n face-dev python=3.9
```

```shell
conda activate face-dev
```

```shell
pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Add new persons to datasets

1. **Create a folder with the folder name being the name of the person**

   ```
   datasets/
   ├── backup
   ├── data
   ├── face_features
   └── new_persons
       ├── name-person1
       └── name-person2
   ```

2. **Add the person's photo in the folder**

   ```
   datasets/
   ├── backup
   ├── data
   ├── face_features
   └── new_persons
       ├── name-person1
       │   └── image1.jpg
       │   └── image2.jpg
       └── name-person2
           └── image1.jpg
           └── image2.jpg
   ```

3. **Run to add new persons**

   ```shell
   python add_persons.py
   ```

4. **Run to recognize**

   ```shell
   python recognize.py
   ```

## Technology

### Face Detection

1. **Retinaface**

   - Retinaface is a powerful face detection algorithm known for its accuracy and speed. It utilizes a single deep convolutional network to detect faces in an image with high precision.

2. **Yolov5-face**

   - Yolov5-face is based on the YOLO (You Only Look Once) architecture, specializing in face detection. It provides real-time face detection with a focus on efficiency and accuracy.

3. **SCRFD**
   - SCRFD (Single-Shot Scale-Aware Face Detector) is designed for real-time face detection across various scales. It is particularly effective in detecting faces at different resolutions within the same image.

### Face Recognition

1. **ArcFace**

   - ArcFace is a state-of-the-art face recognition algorithm that focuses on learning highly discriminative features for face verification and identification. It is known for its robustness to variations in lighting, pose, and facial expressions.

   <p align="center">
   <img src="https://user-images.githubusercontent.com/80930272/160270088-a3760d88-ebc8-4535-907e-6b684276755a.png" alt="ArcFace" />
   <br>
   <em>ArcFace</em>
   </p>

### Face Tracking

1. **ByteTrack**

   <p align="center">
   <img src="./assets/bytetrack.png" alt="ByteTrack" />
   <br>
   <em>ByteTrack is a simple, fast and strong multi-object tracker.</em>
   </p>

### Matching Algorithm

1. **Cosine Similarity Algorithm**

   - The Cosine Similarity Algorithm is employed for matching faces based on the cosine of the angle between their feature vectors. It measures the similarity between two faces' feature representations, providing an effective approach for face recognition.

   <p align="center">
   <img src="https://user-images.githubusercontent.com/80930272/160270156-37fe3269-ca65-4692-a3b2-e9568b3876f8.png" alt="Cosine Similarity Algorithm" />
   <br>
   <em>Cosine Similarity Algorithm</em>
   </p>

## Reference

- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Yolov5-face](https://github.com/deepcam-cn/yolov5-face)
- [InsightFace - ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)
- [InsightFace-REST](https://github.com/SthPhoenix/InsightFace-REST)
