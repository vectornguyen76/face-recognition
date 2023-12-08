# Real-Time Face Recognition

## Development Environment (Ubuntu)

1. **Create Environment and Install Packages**

   ```shell
   conda create -n face-dev python=3.9
   ```

   ```shell
   conda activate face-dev
   ```

   Install lap

   ```
   sudo apt install g++
   ```

   Install OpenCV

   ```
   sudo apt install libgl1-mesa-glx
   ```

   ```shell
   pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```
   python recognize.py
   python tracking.py
   ```

## Reference

- https://github.com/deepcam-cn/yolov5-face
- https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
- https://github.com/SthPhoenix/InsightFace-REST
