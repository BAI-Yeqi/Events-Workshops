# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 16:20:14 2018

combined


"""
import cv2
import sys
from time import sleep
import os
import shutil
import numpy as np
from PIL import Image

### image capture

camera = cv2.VideoCapture(0)

index = 0
expressions_array = ['happy', 'superised', 'wink', 'normal', 'sad', 'angry', 'sleepy', 'laugh']

def get_image():
     # read is the easiest way to get a full image out of a VideoCapture object.
     retval, im = camera.read()
     return im

def get_grp_members():
    grp_size = input("How many people are there in your group?   " )
    members_array = []
    for i in range(1 , int(grp_size) + 1):
        name = input("input the name of member no." + str(i) + ":  " )
        members_array.append(name)
    return members_array

def capturer(index, member, expression):
    while True:
        if not camera.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass
    
        im = get_image()
        
        # Display the resulting frame
        cv2.imshow( member + ', please show the expression: ' + expression, im)
    
        #press "c" to capture
        if cv2.waitKey(1) & 0xFF == ord('c'):
            file = "my captures/" + str(index) + "." + expression + ".jpg"
            
            # A nice feature of the imwrite method is that it will automatically choose the
            # correct format based on the file extension you provide. Convenient!
            cv2.imwrite(file, im)
            cv2.destroyAllWindows()
            break

members_array = get_grp_members()        
        
for i in range(0, len(members_array) ):
    for expression in expressions_array:
        capturer(i, members_array[i], expression)
    
    

# When everything is done, release the capture
camera.release()
cv2.destroyAllWindows()


### image convert


path = './my captures'
image_paths = [os.path.join(path, f) for f in os.listdir(path)]

for image_path in image_paths:
    new_path = image_path.replace('.jpg','')
    # rename: to get rid of '.jpg'
    shutil.move(image_path, new_path)
    

### image recognize
    
    
# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer
# Here modification was made and submodule is used
recognizer = cv2.face.LBPHFaceRecognizer_create()



def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        name = int(os.path.split(image_path)[1].split(".")[0])
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(name)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            # Delay for 50ms
            cv2.waitKey(500)
        
    # return the images list and labels list
    return images, labels

# Path to the Yale Dataset
path = './my captures'
# Call the get_images_and_labels function and get the face images and the 
# corresponding labels
images, labels = get_images_and_labels(path)


cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(images, np.array(labels))

# Append the images with the extension .sad into image_paths
image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad')]
for image_path in image_paths:
    predict_image_pil = Image.open(image_path).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
        nbr_actual = int(os.path.split(image_path)[1].split(".")[0])
        if nbr_actual == nbr_predicted:
            print ("{} is Correctly Recognized with confidence {}".format(members_array[nbr_actual], conf))
        else:
            print ("{} is Incorrect Recognized as {}".format(members_array[nbr_actual], members_array[nbr_predicted]))
        cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
        cv2.waitKey(1000)

# Close the test set window         
cv2.destroyAllWindows()


# empty 'my captures' folder
folder = 'my captures'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)
