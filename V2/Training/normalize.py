import os

from matplotlib import image

height = 0
width = 0
print("running")
for folder in ["train", "test", "val"]:
    files = []
    for r, d, f in os.walk("./data/" + folder):
        #go into each folder
        for file in f:
            if '.png' in file:
                files.append(os.path.join(r, file))
    for f in files:
        # if width and height are 0, set them to the current image's width and height
        if width == 0 and height == 0:
            width, height = image.imread(f).shape[1], image.imread(f).shape[0]
        # now get the corresponding txt file and normalize it
        txt_file = f.replace(".png", ".txt")
        with open(txt_file, "r") as file:
            lines = file.readlines()
        with open(txt_file, "w") as file:
            for line in lines:
                # normalize the coordinates
                line = line.split(" ")
                line[1] = str(float(line[1]) / width)
                line[2] = str(float(line[2]) / height)
                line[3] = str(float(line[3]) / width)
                line[4] = str(float(line[4]) / height)
                file.write(" ".join(line))