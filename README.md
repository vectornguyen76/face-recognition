# Real-Time Face Recognition
<p align="center">
  <img src="./static/results/face-recognition.gif" alt="animated" />
</p>

# How to add 1 peason to recognize
### Step 1: Create a folder with the folder name being the name of the person
### Step 2: Add the person's photo in the folder
### Step 3: Move folder to additional-training-datasets folder
#### Example:

- |database
- ----|additional-training-datasets
- --------|name-person1
- --------|name-person2
- ----|face-datasets
- ----|full-training-datasets

### Step 4: Set up with Python >= 3.7
````
pip install -r requirements.txt
````
### Step 5: Run to add person
````
python train.py --is-add-user=True
````
### Step 6: Run recognize
````
python recognize.py
````
# Face Recognition use Yolov5-face, Insightface, Similarity Measure 
<p align="center">
  <img src="./static/results/workflow.png" alt="animated" />
</p>

# Yolov5-face to dectect face
<p align="center">
  <img src="./static/results/face-detection.gif" alt="animated" />
</p>

# Insight Face to recognition face
![image](https://user-images.githubusercontent.com/80930272/160270088-a3760d88-ebc8-4535-907e-6b684276755a.png)

# Multi thread
<p align="center">
  <img src="https://user-images.githubusercontent.com/80930272/165548024-6d25fbe4-057f-4123-a3f9-3912cce2b73b.png" alt="animated" />
</p>

# Cosine Similarity Algorithm
![image](https://user-images.githubusercontent.com/80930272/160270156-37fe3269-ca65-4692-a3b2-e9568b3876f8.png)

# Reference
- https://github.com/deepcam-cn/yolov5-face
- https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
