import keras
from keras import layers
import cv2
import numpy as np
img=cv2.imread('Chess_Master.jpg') #Reading the image
height,width,channels=img.shape #Defining the shape of the image
print('Height: {}, Width: {}, Channels: {}'.format(height,width,channels))
model=keras.Sequential() #Creating a sequential model
model.add(layers.Input(shape=(height,width,channels)))
model.add(layers.Dense(32)) #Adding a dense layer with 32 neurons
model.add(layers.Dense(16)) #Adding a dense layer with 16 neurons
model.add(layers.Dense(2)) #Adding a dense layer with 2 neurons
preprocessed_img=np.array([img]) #We are passing it as a numpy array
result=model(preprocessed_img) #Passing the numpy array through the model
print(result) #Printing the result array
cv2.imshow('img',img) #Displaying the image
cv2.waitKey(0)
cv2.destroyAllWindows()