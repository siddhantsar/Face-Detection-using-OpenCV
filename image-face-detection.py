import cv2
import csv

if __name__ == "__main__":
	# Loading the cascade XML file using CascadeClassifier method. 
	face_cascades = cv2.CascadeClassifier("xml-files/haarcascade_frontalface_alt.xml")

	img = cv2.imread("images/image-group-1.jpg")
	img = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation = cv2.INTER_CUBIC)

	# For storing x, y pixels of the face in any given image
	faces = face_cascades.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5)

	for x, y, w, h in faces:
		# To draw rectamgle, args: img, (starting point corner), (end point corner), (color), width)
		img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

	cv2.imshow("Output", img)
	cv2.waitKey(0)

