import cv2
import datetime as dt
from time import sleep

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
f= open("user numbers log.txt","w+")


video_capture = cv2.VideoCapture(0)
user_no_prev = 0
time_prev = dt.datetime.now()

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    ###added by BYQ
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
    ###BYQ peace out


    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

  
    #log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)

    #press "q" to quit app
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

f.close()
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
