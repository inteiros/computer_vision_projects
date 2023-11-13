# Webcam Capture Tracker and Face detection

Quick motion tracker and face detection scipts using opencv and haarcascade_frontalface_default classifier

## Running the scripts

First off, install opencv with pip

```bash
pip install cv2
```

Now, you can run the script with

```bash
python facedetection.py
```

and the script will detect faces with the camera

![demo](facedetection.gif "demo")

or

```bash
python tracker.py
```

The script will render a frozen frame so you can select the object with your mouse and press enter or space to proceed (or press c to cancel and select the object again). Quit the application by pressing Q on your keyboard.

![demo](objtracker.gif "demo")
