import os

# get all txt files and remove the go back to line
#the structure is :
#train
## labels
### 1
#### 1.01.txt
#### 1.02.txt
#### 1.03.txt
### 2
#### 2.01.txt
#### 2.02.txt
files = []
print("running")
for folder in ["test", "train", "val"]:
    for r, d, f in os.walk("./data/"+folder):
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
                if count == 0:
                    tclass = line.replace("\n", " ")
                    file.write(tclass)
                elif count != 1:
                    file.write(tclass + line)
                else:
                    file.write(line) 
                count += 1
