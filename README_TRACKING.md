## Change logs:
1. Add ```recognize-camera.py```: camera ver. of recognize.py
2. Add ```tracking-camera.py```: tracking with live camera & multithreading

## Notes:
1. Here I used [IP Camera](https://play.google.com/store/apps/details?id=com.pas.webcam) - Android Live IP Camera. 
2. You can modify the code to change the input from IP Camera to your computer's camera. (there is ```Camera()``` function in ```tracking-camera.py```)


## Run
Install env packages:

```pip install -r requirements.txt```


Install tracking (ByteTrack) and other packages: Checkout [face_tracking](.face_tracking\README.md)

Run: ```python tracking-camera.py```
