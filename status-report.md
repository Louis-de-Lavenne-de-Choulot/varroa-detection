# Varroa Detection
## status report - 27/02/2023

The goal of this project is to develop an AI system capable of detecting the parasitic mite Varroa on bees and producing a report to alert beekeepers using long distance signals. I have decided to move forward with TensorFlow and Kaggle for development as I have previous experience with TensorFlow. I have trained a basic model using a dataset found on the Ib after extensive research.

## Updates


### Model Progress
I have made significant progress in training the model to detect Varroa on bees. However, I am currently facing a challenge where the model produces a size error at the entry point of the model when the image sizes vary from the dataset pictures. I am working on solving this issue by exploring different methods of resizing and normalization.

### Real-time AI Library
 I have recently discovered a real-time AI library named YOLO, which has the potential to improve the system's accuracy and efficiency while minimizing CPU usage. I am currently evaluating YOLO's performance to determine if it is a viable solution for the project.

### Continuous training
 I am exploring the idea of continuous training implementation to allow the user to upgrade his AI via feedback to better match the local environment.

### Data Augmentation
 I am also exploring the use of data augmentation techniques to increase the size and diversity of the training dataset. This approach can help to improve the model's ability to generalize and identify Varroa in a variety of settings and conditions. It has been quite difficult to find datasets of bees with varroa so I am contemplating two different options, either to find a way to zoom in and find the same resolution as the one of the current dataset or continue to search for better datasets.

### Users
I have one beekeeper that allowed the test on his beehives once the product is realised as long as no harm is done to the bees. The problem is: will there be varroa to detect on any of his beehives ? I am searching for other beekeepers to extend the testing. For now any real life testing would at least permit to test if no false alarm is raised or if the bees attack the device.

### Current and next Steps
I am making steady progress towards the goal of developing the AI system for detecting Varroa on bees. I will continue to explore new techniques and solutions to improve the accuracy and efficiency of the model. Once the AI is able to detect through a camera, the next step will be the hardware.