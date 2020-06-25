import sys
import cv2

# Get data for face detection
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Get data for left eye detection
face_cascade_left_eye = cv2.CascadeClassifier('data/haarcascade_lefteye_2splits.xml')

# Get data for right eye detection
face_cascade_right_eye = cv2.CascadeClassifier('data/haarcascade_righteye_2splits.xml')

# Get Camera
video_capture = cv2.VideoCapture(0)
video_capture.set(10, 120)

# Create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')

#Save video
out = cv2.VideoWriter('records/output.avi', fourcc, 20.0, (640, 480))

while True:

    # Capture frame by frame
    retval, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(25, 30)
    )

    face_left_eye = face_cascade_left_eye.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(25, 25)
    )

    face_right_eye = face_cascade_right_eye.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(25, 25)
    )

    # Draw rectangle around face
    for (x, y, w, h) in faces:
        face_image = cv2.rectangle(frame, (x, y), (x+w, y+h), (176, 4, 179), 2)
        cv2.putText(face_image, 'face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 217, 210), 2)

        # Draw rectangle around left eye
        for (xe, ye, we, he) in face_left_eye:
            cv2.rectangle(frame, (xe, ye), (xe+we, ye+he), (179, 7, 4), 2)

        # Draw rectangle around right eye
        for (xr, yr, wr, hr) in face_right_eye:
            cv2.rectangle(frame, (xr, yr), (xr+wr, yr+hr), (179, 7, 4), 2)


    # Show every frame
    cv2.imshow('Video', frame)

    # Write to frame from output
    out.write(frame)

    # If q is pushed, exit programm
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit()
