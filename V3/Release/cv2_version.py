import io
import os
import sys
import subprocess
import re
import time
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk
import torch
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5.utils.torch_utils import select_device, smart_inference_mode

# Static variables from code 1
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
project = ROOT / "runs/run"  # save results to project/name4
name = 'run'  # save results to project/name
exist_ok = False  # existing project/name ok, do not increment
captures_folder = "./captures"  # Folder to store captured images
save_img = True  # Always save inference images
save_txt = True  # save results to *.txt
data = ROOT / 'data/coco128.yaml'  # dataset.yaml path
# Directories
save_dir = increment_path(Path(project), exist_ok=exist_ok)  # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

# Function to perform single-image detection (from code 1)
def detect_single_image(weights, source, imgsz, conf_thres, iou_thres, max_det, device):
    bs = 1  # batch size
    # Load YOLOv5 model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
     # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        print(im.shape)
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0  # for save_crop
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\\n')

                    if save_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label =  f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

# Function to capture an image from the camera and perform prediction (from code 2)
def capture_and_process_images(interval=10):
    image_count = 0
    # while True:
    if True:
        # Capture an image using ffmpeg and save it with a unique filename
        timestamp = time.strftime("%Y%m%d%H%M%S")  # Generate a timestamp
        image_name = f"image_{timestamp}_{image_count}.jpg"
        image_path = os.path.join(captures_folder, image_name)
        
        # Read the frame from the video capture device
        ret, frame = cap.read()
        # if not ret:
        #     continue
        cv2.imwrite(image_path, frame)

        # Perform image detection
        detect_single_image(weights, image_path, imgsz, conf_thres, iou_thres, max_det, device)

        # Increment image_count for the next image
        image_count += 1

        # Sleep for the specified interval (in seconds)
        time.sleep(interval)

if __name__ == '__main__':
    # Static variables from code 1
    weights = './ai_model/best_100.pt'
    imgsz = [416, 416]
    conf_thres = 0.6
    iou_thres = 0.45
    max_det = 1000
    device = ''

    # Initialize the VideoCapture object
    cap = cv2.VideoCapture(0)
    # Check if the VideoCapture object is   initialized successfully
    if not cap.isOpened():
        raise ValueError("Failed to open the video capture device")

from flask import Flask, render_template, Response, request

app = Flask(__name__)

# Camera Settings
gain = 174
cap.set(cv2.CAP_PROP_GAIN, gain)
brightness = 50
cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
contrast = 67
cap.set(cv2.CAP_PROP_CONTRAST, contrast)
saturation = 85
cap.set(cv2.CAP_PROP_SATURATION, saturation)
exposure = -6
cap.set(cv2.CAP_PROP_EXPOSURE, exposure)


def adjust_camera_settings(gain, brightness, contrast, saturation, exposure):
    cap.set(cv2.CAP_PROP_GAIN, int(gain))
    cap.set(cv2.CAP_PROP_BRIGHTNESS, int(brightness))
    cap.set(cv2.CAP_PROP_CONTRAST, int(contrast))
    cap.set(cv2.CAP_PROP_SATURATION, int(saturation))
    cap.set(cv2.CAP_PROP_EXPOSURE, int(exposure))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        gain = request.form['gain']
        brightness = request.form['brightness']
        contrast = request.form['contrast']
        saturation = request.form['saturation']
        exposure = request.form['exposure']
        adjust_camera_settings(gain, brightness, contrast, saturation, exposure)# Get the current camera settings
    gain = cap.get(cv2.CAP_PROP_GAIN)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    saturation = cap.get(cv2.CAP_PROP_SATURATION)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)

    return render_template('index.html', gain=int(gain), brightness=int(brightness), contrast=int(contrast),
   saturation=int(saturation), exposure=int(exposure))


# Function to generate video frames
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize the frame to fit the window
        frame_resized = cv2.resize(frame_rgb, (640, 480))

        # Convert the frame to PIL Image format
        pil_image = Image.fromarray(frame_resized)
        # Convert PIL Image to byte array
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # Yield the byte array as a video frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_byte_arr + b'\r\n')


# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4025)

# Release the VideoCapture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()