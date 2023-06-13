'''
Trash Detection
Designed to run on a Raspberry Pi 4 running Raspberry Pi OS (64-bit) and using a Raspberry Pi Camera
The trash_model.pt runs on yolov5 
'''

# Imports
import cv2
import torch
from picamera2 import Picamera2
import time
import os

# Load the trash detection model
print('Loading Model...')
model_path = 'trash_model.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Initialize the Pi Camera
pi_camera = Picamera2()
# Convert the color mode to RGB
config = pi_camera.create_preview_configuration(main={"format": "RGB888"})
pi_camera.configure(config)

# Start the pi camera and give it a second to set up
pi_camera.start()
time.sleep(1)

def detect_objects(image):
	'''
	Returns a list of tuples for each object detected
	the tuple is consists of:
		(x,y) coordinate of the top left coner of the bouding box of the object detected as an int tuple
		(x,y) coordinate of the lower right coner of the bouding box of the object detected as an int tuple
		the name of the object detected as a string
		the confidence of the object detected as a float	

	if no objects are detected an empty list is returned	
	'''
	print('Detecting Trash...')

	# Save the image temporally 
	temp_path = 'temp.png'
	cv2.imwrite(temp_path, image)

	# Create a list of image for the model to process
	imgs = [temp_path]

	# Run the model
	results = model(imgs)
	df = results.pandas().xyxy[0]

	# Create a list to hold the detected objects
	detected_objects = []

	# Go through each objected that was detected and add it to the list
	for index, row in df.iterrows():
		# Save the starting point and ending point of the bounding box for the object detected
		# Save the name of the object detected
		# Save the confidence object detected
		p1 = (int(row['xmin']), int(row['ymin']))
		p2 = (int(row['xmax']), int(row['ymax']))
		name = row['name']
		confidence = round(row['confidence'] * 100, 2)

		detected_objects.append((name, p1, p2, confidence))

	# delete the image file
	os.remove(temp_path)

	return detected_objects


def draw_on_image(image, objectsDetected, color=(255, 0, 0), thickness=2, fontScale=1, font=cv2.FONT_HERSHEY_SIMPLEX):
	'''
	Draws a bounding box and the name of the object detected on the image
	'''
	print('Drawing on Image...')
	for name, start_point, end_point, confidence in objectsDetected:
		# Draw the bounding box
		image = cv2.rectangle(image, start_point, end_point, color, thickness)

		# Add Text to the bounding box
		org = (start_point[0], start_point[1] - 10)
		image = cv2.putText(image, f'{name} {confidence}%', org, font, fontScale, color, thickness, cv2.LINE_AA)
	return image


def main():
	while True:
		# Get a image frame as a numpy array
		image = pi_camera.capture_array()

		# Detect Objects
		objectsDetected = detect_objects(image)

		# Draw on the image
		image = draw_on_image(image, objectsDetected)

		# display the image
		cv2.imshow("Video", image)

		# This waits for 1 ms and if the 'q' key is pressed it breaks the loop	 
		if cv2.waitKey(42) == ord('q'):
			break


if __name__ == "__main__":
	main()
	print('Done!!')