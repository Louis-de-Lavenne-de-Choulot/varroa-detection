import os
from PIL import Image

files = []
for r, d, f in os.walk("./data"):
    # Get all PNG files and convert to JPG
    for file in f:
        if '.png' in file:
            files.append(os.path.join(r, file))

for f in files:
    # Open the PNG image
    png_image = Image.open(f)
    
    # Replace '.png' with '.jpg' in the filename
    jpg_filename = f.replace(".png", ".jpg")
    
    # Convert and save as JPG
    jpg_image = png_image.convert("RGB")
    jpg_image.save(jpg_filename, "JPEG")
    
    # Remove the original PNG file
    os.remove(f)
