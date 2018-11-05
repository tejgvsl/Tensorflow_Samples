import cv2     # for capturing videos
import math   # for mathematical operations
import os
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
Test_path = './Image_frames/Test_frames'
Train_path = './Image_frames/Train_frames'
count = 0
videoFile = "Tom and jerry.mp4"
cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
frameRate = cap.get(cv2.CAP_PROP_FPS) #frame rate
print('Frame rate = ' , frameRate)

while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename ="frame%d.jpg" % count;count+=1
        cv2.imwrite(os.path.join(Train_path , filename), frame)
cap.release()
print ("############## Train Video reading is Done!!!! ##########")

#img = cv2.imread('./Image_frames/frame0.jpg')   # reading image using its name
#cv2.imshow('Frame',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

data = pd.read_csv('mapping.csv')     # reading the csv file
print(data.head(15))      # printing first five rows of the file
print ("CSV reading is Done!!!! ")

#Our next step is to read the images which we will do based on their names, aka, the Image_ID column.
X = [ ]     # creating an empty array
for img_name in data.Image_ID:
    img = plt.imread(os.path.join(Train_path , img_name))
    X.append(img)  # storing each image in array X
X = np.array(X)    # converting list to array

#Since there are three classes, we will one hot encode them using the to_categorical() function of keras.utils.
y = data.Class
dummy_y = np_utils.to_categorical(y)    # one hot encoding Classes
print(dummy_y.size)
print(dummy_y.shape)

#We will be using a VGG16 pretrained model which takes an input image of shape (224 X 224 X 3).
#Since our images are in a different size, we need to reshape all of them.
#We will use the resize() function of skimage.transform to do this.
image = [ ]
for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    image.append(a)
image = np.array(image)

#All the images have been reshaped to 224 X 224 X 3. But before passing any input to the model,
#we must preprocess it as per the model’s requirement. Otherwise, the model will not perform well enough.
#Use the preprocess_input() function of keras.applications.vgg16 to perform this step.
from keras.applications.vgg16 import preprocess_input
X = preprocess_input(image, mode='tf')      # preprocessing the input data

#We also need a validation set to check the performance of the model on unseen images.
#We will make use of the train_test_split() function of the sklearn.model_selection module to randomly
#divide images into training and validation set.
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)    # preparing the validation set

#The next step is to build our model. As mentioned, we shall be using the VGG16 pretrained model for this task.
#Let us first import the required libraries to build the model:
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout

#We will now load the VGG16 pretrained model and store it as base_model:
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    # include_top=False to remove the top layer

#We will make predictions using this model for X_train and X_valid, get the features,
#and then use those features to retrain the model.
X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
X_train.shape, X_valid.shape

#The shape of X_train and X_valid is (208, 7, 7, 512), (90, 7, 7, 512) respectively.
#In order to pass it to our neural network, we have to reshape it to 1-D.
X_train = X_train.reshape(208, 7*7*512)      # converting to 1-D
X_valid = X_valid.reshape(90, 7*7*512)

#We will now preprocess the images and make them zero-centered which helps the model to converge faster.
train = X_train/X_train.max()      # centering the data
X_valid = X_valid/X_train.max()

#Finally, we will build our model. This step can be divided into 3 sub-steps:
#1. Building the model
#2. Compiling the model
#3. Training the model

#Building the model
model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='sigmoid')) # hidden layer
model.add(Dense(3, activation='sigmoid'))    # output layer

#printing Model summary
model.summary()

#We have a hidden layer with 1,024 neurons and an output layer with 3 neurons (since we have 3 classes to predict).
#Now we will compile our model:
#Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))

print("##########  Trainig is Done  ############")

#Calculating the screen time – A simple solution
#First, download the video we’ll be using in this section from here.
#Once done, go ahead and load the video and extract frames from it.
#We will follow the same steps as we did above:
count = 0
videoFile = "Tom and Jerry 3.mp4"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(cv2.CAP_PROP_FPS) #frame rate
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename ="test%d.jpg" % count;count+=1
        cv2.imwrite(os.path.join(Test_path , filename), frame)
cap.release()
print ("********* Reading test video is Done! ********")

#After extracting the frames from the new video,
#we will now load the test.csv file which contains the names of each extracted frame.
#Download the test.csv file and load it:
test = pd.read_csv('test.csv')
print ("********* Reading test CSV is Done! ********")
#Next, we will import the images for testing and then reshape them
#as per the requirements of the aforementioned pretrained model:
test_input = [ ]
for img_name in test.Image_ID:
    img = plt.imread(os.path.join(Test_path , img_name)) #plt.imread('' + img_name)
    test_input.append(img)
test_input = np.array(test_input)

test_image = [ ]
for i in range(0,test_input.shape[0]):
    a = resize(test_input[i], preserve_range=True, output_shape=(224,224)).astype(int)
    test_image.append(a)
test_image = np.array(test_image)

#We need to make changes to these images similar to the ones
#we did for the training images. We will preprocess the images,
#use the base_model.predict() function to extract features from these images
#using the VGG16 pretrained model, reshape these images to 1-D form, and make them zero-centered:

# preprocessing the images
test_image = preprocess_input(test_image, mode='tf')

# extracting features from the images using pretrained model
test_image = base_model.predict(test_image)

# converting the images to 1-D form
test_image = test_image.reshape(186, 7*7*512)

# zero centered images
test_image = test_image/test_image.max()

#Since we have trained the model previously, we will make use of that model to make prediction for these images
predictions = model.predict_classes(test_image)

#Calculate the screen time of both TOM and JERRY
#Recall that Class ‘1’ represents the presence of JERRY, while Class ‘2’ represents the presence of TOM.
#We shall make use of the above predictions to calculate the screen time of both these legendary characters:

print("The screen time of JERRY is", predictions[predictions==1].shape[0], "seconds")
print("The screen time of TOM is", predictions[predictions==2].shape[0], "seconds")
