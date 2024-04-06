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
    # Check if the VideoCapture object is initialized successfully
    if not cap.isOpened():
        raise ValueError("Failed to open the video capture device")


#Camera Settings
gain=174
cap.set(cv2.CAP_PROP_GAIN,gain)

brightness=50
cap.set(cv2.CAP_PROP_BRIGHTNESS,brightness)

contrast=67
cap.set(cv2.CAP_PROP_CONTRAST,contrast)

saturation=85
cap.set(cv2.CAP_PROP_SATURATION,saturation)

exposure=-9
cap.set(cv2.CAP_PROP_EXPOSURE,exposure)

# Function to adjust the camera settings
def adjust_camera_settings():
    # Get the current camera properties
    gain = cap.get(cv2.CAP_PROP_GAIN)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    saturation = cap.get(cv2.CAP_PROP_SATURATION)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)

    # Open a settings window
    settings_window = tk.Toplevel(root)
    settings_window.title("Camera Settings")

    # Function to update the camera settings
    def update_settings():
        new_gain = gain_scale.get()
        new_brightness = brightness_scale.get()
        new_contrast = contrast_scale.get()
        new_saturation = saturation_scale.get()
        new_exposure = exposure_scale.get()
        cap.set(cv2.CAP_PROP_GAIN, new_gain)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, new_brightness)
        cap.set(cv2.CAP_PROP_CONTRAST, new_contrast)
        cap.set(cv2.CAP_PROP_SATURATION, new_saturation)
        cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)

    # Create a gain scale
    gain_label = tk.Label(settings_window, text="Gain")
    gain_label.pack()
    gain_scale = tk.Scale(settings_window, from_=0, to=255, resolution=1, orient=tk.HORIZONTAL)
    gain_scale.set(gain)
    gain_scale.pack()

    # Create a brightness scale
    brightness_label = tk.Label(settings_window, text="Brightness")
    brightness_label.pack()
    brightness_scale = tk.Scale(settings_window, from_=0, to=255, resolution=1, orient=tk.HORIZONTAL)
    brightness_scale.set(brightness)
    brightness_scale.pack()

    # Create a contrast scale
    contrast_label = tk.Label(settings_window, text="Contrast")
    contrast_label.pack()
    contrast_scale = tk.Scale(settings_window, from_=0, to=255, resolution=1, orient=tk.HORIZONTAL)
    contrast_scale.set(contrast)
    contrast_scale.pack()

    # Create a saturation scale
    saturation_label = tk.Label(settings_window, text="Saturation")
    saturation_label.pack()
    saturation_scale = tk.Scale(settings_window, from_=0, to=255, resolution=1, orient=tk.HORIZONTAL)
    saturation_scale.set(saturation)
    saturation_scale.pack()

    # Create an exposure scale
    exposure_label = tk.Label(settings_window, text="Exposure")
    exposure_label.pack()
    exposure_scale = tk.Scale(settings_window, from_=-10, to=10, resolution=1, orient=tk.HORIZONTAL)
    exposure_scale.set(exposure)
    exposure_scale.pack()

    # Create an update button
    update_button = tk.Button(settings_window, text="Update", command=update_settings)
    update_button.pack()

# Rest of the code...
# Create the main window
root = tk.Tk()

# Create a label to display the video feed
label = tk.Label(root)
label.pack()

# Create a button to capture and process images
capture_button = tk.Button(root, text="Capture", command=capture_and_process_images)
capture_button.pack()

# Create a button to adjust camera settings
settings_button = tk.Button(root, text="Settings", command=adjust_camera_settings)
settings_button.pack()

# Function to update the video feed
def update_video_feed():
    ret, frame = cap.read()
    if ret:
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize the frame to fit the window
        frame_resized = cv2.resize(frame_rgb, (640, 480))
        # Create an ImageTk object from the resized frame
        img = ImageTk.PhotoImage(Image.fromarray(frame_resized))
        # Update the label with the new image
        label.configure(image=img)
        label.image = img

    # Schedule the next update after 10 milliseconds
    root.after(1, update_video_feed)

# Start updating the video feed
update_video_feed()

# Start the Tkinter event loop
root.mainloop()

# Release the VideoCapture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
