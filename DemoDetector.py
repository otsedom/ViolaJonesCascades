import cv2  # OpenCV Library

# -----------------------------------------------------------------------------
#       Load and configure Haar Cascade Classifiers
# -----------------------------------------------------------------------------

# location of OpenCV Haar Cascade Classifiers:
baseCascadePath = './Cascades/'

# xml file describing our haar cascade classifier to use
CascadeFilePath = baseCascadePath + 'haarcascade_mcs_upperbody.xml'

# build our cv2 Cascade Classifiers
Cascade = cv2.CascadeClassifier(CascadeFilePath)

# -----------------------------------------------------------------------------
#       Main program loop
# -----------------------------------------------------------------------------

# collect video input from first webcam on system
video_capture = cv2.VideoCapture(0)

while True:
    # Capture video feed
    ret, frame = video_capture.read()

    # Create greyscale image from the video feed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect object in input video stream
    objects = Cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Show detections
    print "Found {0} objects!".format(len(objects))

    # Iterate over each face found
    for (x, y, w, h) in objects:
    # Draw object containers
        container = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # press any key to exit
    # NOTE;  x86 systems may need to remove: &amp;amp;amp;amp;amp;amp;quot;&amp;amp;amp;amp;amp;amp;amp; 0xFF == ord('q')&amp;amp;amp;amp;amp;amp;quot;
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()