# Real Time 2D Object Detection 

The objective of this project was to develop a real time 2D object recognition system, which can recognise a set of objects in a rotation, scale and translation invariant manner. This was achieved by a series of steps, beginning with converting the image or frame into a binary image using thresholding and filling in holes and removing unwanted noise, then finding the required region containing the object. This was followed by computing feature vectors of the objects and storing them in a database to be used as a training model. Then for the final step, it was required to develop a recognition system with 2 different classifiers. The final model is able to classify 15 objects in real time and also has the capability to detect unknown objects and gives the user the capability to train the model for the unknown object. 

Wiki Khoury link : https://wiki.khoury.northeastern.edu/display/~kishore005/Real+Time+2D+Object+Detection

Compile using cmake file.
1. cmake .
2. make
3. ./CV_PROJECT_3
