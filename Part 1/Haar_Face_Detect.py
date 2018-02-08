import cv2

#check version of library:
#see: [https://stackoverflow.com/questions/20180543/how-to-check-version-of-python-modules]
print ("openCV version:"+cv2.__version__)

# Get user supplied values
imagePath = "./abba.png"
cascPath = "./haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
#need to study this function when preparing the 
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    #a modification is needed:
    #see: [https://stackoverflow.com/questions/36242860/attribute-error-while-using-opencv-for-face-recognition]
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
# waitKey(0) will display the window infinitely until any keypress (it is suitable for image display)
cv2.waitKey(0)
