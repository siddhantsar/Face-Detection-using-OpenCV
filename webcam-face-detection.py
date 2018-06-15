import cv2
import numpy as np 

# Loading the cascade xml file
face_cascades = cv2.CascadeClassifier("xml-files/haarcascade_frontalface_alt.xml")

# Intializing the webcam
cap = cv2.VideoCapture(0)
scaling_factor = 0.5
print("Press Esc to exit.")

if __name__ == "__main__":
	while True:
	    ret, frame = cap.read()
	    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

	    # Detecting faces within frame and getting the pixel values
	    faces = face_cascades.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3)

	    for (x, y, w, h) in faces:
	    	# Drawing a rectangle around a face detected
	        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

	    cv2.imshow("Face Detector", frame)

	    c = cv2.waitKey(1)
	    if c==27:
	        break

	cap.release()
	cv2.destroyAllWindows()
