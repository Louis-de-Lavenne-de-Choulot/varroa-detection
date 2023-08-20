from pathlib import Path
import torch



model ="./choosen_model/best1.pt"

# img can be taken from stream
source_image_path = Path('imgpath')
img_size = 416
# 0.6 is the most efficient to sort out the false positive
conf_threshold = 0.6

# command python ./V2/content/yolov5/detect.py --weights model --img img_size --conf conf_threshold --source img --save-txt  
import subprocess
subprocess.run(['python', './V2/content/yolov5/detect.py', '--weights', model, '--img', img_size, '--conf', conf_threshold, '--source', source_image_path, '--save-txt']) 
