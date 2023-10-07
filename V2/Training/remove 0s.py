
import os

files = []
print("running")
for folder in ["train", "test", "val"]:
    for r, d, f in os.walk("./data/" + folder):
        #go into each folder
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))
    for f in files:
        with open(f, "r") as file:
            lines = file.readlines()
        with open(f, "w") as file:
            tclass = ""
            count = 0
            for line in lines:
                # if lines contains a 0, remove it
                if line[0] == "0" and count == 0:
                    file.write("")
                    count += 1
                    break
                else:
                    file.write(line)
