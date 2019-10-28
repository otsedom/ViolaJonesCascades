import cv2  # OpenCV Library

# ---------------------------------------------------------------------------------------------------------
#       Load and configure Haar Cascade Classifiers, searching nses only inside face containers
# ---------------------------------------------------------------------------------------------------------

# location of OpenCV Haar Cascade Classifiers:
baseCascadePath = './Cascades/'

# xml files describing our haar cascade classifiers
# Default opencv face detector
faceCascadeFilePath = baseCascadePath + 'haarcascade_frontalface_default.xml'
noseCascadeFilePath = baseCascadePath + 'haarcascade_mcs_nose.xml'

# build our cv2 Cascade Classifiers
faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
noseCascade = cv2.CascadeClassifier(noseCascadeFilePath)

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

    # Detect faces in input video stream
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Show detections
    if len(faces)==1:
        print("Found {0} face!".format(len(faces)))
    else:
        print("Found {0} faces!".format(len(faces)))

    # Iterate over each face found
    for (x, y, w, h) in faces:
    # Un-comment the next line for debug (draw box around all faces)
        face = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        #Searching noses in the lower 2/3 of the face
        roi_gray = gray[y + (int)(h/3):y + h, x:x + w]
        roi_color = frame[y + (int)(h/3):y + h, x:x + w]

         # Detect a nose within the region bounded by each face (the ROI)
        nose = noseCascade.detectMultiScale(roi_gray)

        for (nx, ny, nw, nh) in nose:
            # Un-comment the next line for debug (draw box around the nose)
             cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),2)

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