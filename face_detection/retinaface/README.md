<h1>Get weights: </h1>
[Google drive](https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1?usp=drive_link)

<h1>Run</h1>

<h3> Using Camera </h3>
backbone: resnet50

```
python camera_test.py --trained_model weights/Resnet50_Final.pth --network resnet50 --cpu
```

backbone: mobilenet0.25

```
python camera_test.py --trained_model weights/mobilenet0.25_Final.pth --network mobile0.25 --cpu
```

<h3> Using Image </h3>
change image in ./curve, change file path in detect.py (line 87)

backbone: resnet50

```
python detect.py --trained_model weights/Resnet50_Final.pth --network resnet50 --cpu
```

backbone: mobilenet0.25

```
python detect.py --trained_model weights/mobilenet0.25_Final.pth --network mobile0.25 --cpu
```
