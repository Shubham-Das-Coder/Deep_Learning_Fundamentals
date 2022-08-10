import cv2
import keras
from keras.layers import Dense
img=cv2.imread('Chess_Master.jpg',cv2.IMREAD_GRAYSCALE) #reading the image and converting it to a grayscale image
height,width=img.shape #Defining the shape of the image
print('Img shape',img.shape) #Printing the shape of the image
input_layer=keras.Input(shape=(height,width)) #Creating the input layer with with the shape of the image
print('Input layer shape:',input_layer.shape) #Displaying the shape of the input layer
layer_1=Dense(64)(input_layer) #Creating the first hidden layer with 64 neurons and connecting the input_layer 
layer_2=Dense(32)(layer_1) #Creating the second hidden layer with 32 neurons and connecting the layer_1
output_layer=Dense(2)(layer_2) #Creating the output layer with 2 neurons and connecting the layer_2
model=keras.Model(inputs=input_layer,outputs=output_layer) #Defining the model and providing the input_layer and the output_layer
model.summary() #Getting the structure of the created model
cv2.imshow('Chess_Master',img)
cv2.waitKey()
cv2.destroyAllWindows()