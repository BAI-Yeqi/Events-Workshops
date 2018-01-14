
# Part 2: Create a user log file with the camera

## Programming Exercise

#### 1. Import the modules needed


```python
import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
```

#### 2. assign the path of the knowledge file (as a string value) to a variable called 'cascPath'

> string value: "haarcascade_frontalface_default.xml"

#### 3. Create an instance of the CascadeClassifier, assign it to a variable called 'faceCascade'

> to create an instance of the classifier, call the method cv2.CascadeClassifier(), where 'cascPath' is the input.

#### 4. Create a file object named 'f' , with file name 'user numbers log.txt' and mode 'w+'


<img src="https://github.com/BAI-Yeqi/UAVionics_IET_ML_Workshop/blob/master/imgs/fileIO.JPG" width='100%'>

reference: https://docs.python.org/3/tutorial/inputoutput.html
> 'w+'

> Opens a file for both writing and reading. Overwrites the existing file if the file exists. If the file does not exist, creates a new file for reading and writing.

#### 5. create a video Capture object and name it as video_capture

>To capture a video, you need to create a VideoCapture object. 

>Find your answer in the following webpage: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

#### 6. initialize an integer called 'user_no_prev' with value 0 and a datetime object called 'time_prev'

> 'time_prev' will save the current local data and time (the exact time ogf this moment)

> find your answer in the following page: https://docs.python.org/3/library/datetime.html#datetime.datetime


```python


```

> the values above will be compared with real-time values in the loop, so that we can tell if no. of people that appears in the camera has changed, in every loop.

#### 7. Now we have prepared all the objects/variables needed for the "user_log_maker", the next thing we will do is to build a loop which will:


<font color = 'purple'>
<p>
<br>
(a) repeatedly check the availability of the camera;
<br><br>
(b) repeatedly read image from the camera video stream, detect faces in the stream;
<br><br>
(c) repeatedly figure out whether the number of faces appeared in the camera has changed, if the answer is 'yes', record the picture and mark it down in the log file;
<br><br>
(d) using the data returned by (b), repeatedly (or in another word, continously) display the image (which will form a video stream on your screen) with rectangles that highlight the faces.
<br>
</p>
</font>

#### we will build the loop in a modular way, let us now prepares the components needed for the loop:

(a) repeatedly check the availability of the camera ------>  function check_camera()


```python
def check_camera():
     if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass
```

(b) repeatedly read image from the camera video stream, detect faces in the stream

> Call the 'read()' method of your video_capture object, store the outputs as 'ret' (which indicates whether the image reading is successful), 'frame' (the numpy array version of the video frame)

> Convert 'frame' into Gray Scale, and store it as 'gray'

> Call the 'detectMultiscale()' method

<font color = 'pink'> Hint: do refer to the Part 1 step by step tutorial </font>


```python
def detect_faces():
    
    ### Start your coding here:
    

    
    
    
    
    
    
    
    
    
    
    ### End of your coding
    
    return faces,frame
```

(d) using the data returned by (b), repeatedly (or in another word, continously) display the image (which will form a video stream on your screen) with rectangles that highlight the faces. 

>   1. draw the rectangle
>   2. display the image

<font color = 'pink'> Hint: do refer to the Part 1 step by step tutorial </font>


```python
def video_player():
    # Draw a rectangle around the faces
       
        
        
    # Display the resulting frame
    cv2.imshow('Video', frame)
```

(c) repeatedly figure out whether the number of faces appeared in the camera has changed, if the answer is 'yes', record the picture and mark it down in the log file 

> 1 call the 'check_camera()' function

> 2 call the 'detect_faces()' function, assign the outputs to teo variables called 'faces' and 'frame'


<font size = 4> Now, complete the code below </font>


```python
while True:
    
    ### start your code here (about 2 lines of code)   
 


    ### end of your code

    
        
    ### BYQ helped you code this part already
    ### when no. of users appeared in the camera changes, record into the log file and take a picture   
    user_no = len(faces)
    if user_no != user_no_prev:
        f.write("time duration: from " + str(time_prev) + " to " + str(dt.datetime.now()) + "\n")
        f.write("      there are %d users  \n" % user_no)
        if user_no_prev == 0:
            log_frame = frame
        if user_no_prev > 0:
            photo_name = "from_" + str(time_prev.strftime("%H_%M_%S")) + "_to_" + str(dt.datetime.now().strftime("%H_%M_%S"))
            cv2.imwrite(photo_name + '.png',log_frame)
            
    user_no_prev = user_no    
    time_prev = dt.datetime.now()  
    ### BYQ peace out

    
    
    ### start your code here ( 1 line )
    ### display the video, with the funtion we built
    

    
    ### end of your code
    
    
     
    # press "q" to quit app
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


```

<font color='red ' size = 4> Type "Q" to quit the loop </font>

#### 8. Close your file object, release your camera and destroy the video player window
 

    > ### Hint:
    "name of the file object".close()
    "name of the video capture object".release()
    "name of the computer vision package we imported".destroyAllWindows()


```python
### start your code here (about three lines of code)
 
    
    
### end of your code
```
