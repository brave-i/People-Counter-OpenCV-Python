from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


#DEFINE SOME PARAMATERS:
DETECT_CONFIDENCE = 0.4
SKIP_FRAME_SIZE = 20

MODEL_TXT_FILE = "MODEL/SSD_MODEL.txt"
MODEL_BIN_FILE = "MODEL/SSD_MODEL.bin"

INPUT_FILE_NAME = "input_video/input.mp4"
EXPORT_FILE_NAME = "export_video/export.avi"

net = cv2.dnn.readNetFromCaffe(MODEL_TXT_FILE, MODEL_BIN_FILE)
vs = cv2.VideoCapture(INPUT_FILE_NAME)

writer = None
W = None
H = None

totalFrames = 0
totalDown = 0
totalUp = 0

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

fps = FPS().start()
while True:
	frame = vs.read()
	frame = frame[1]

	if INPUT_FILE_NAME is not None and frame is None:
		break

	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	if W is None or H is None:
		(H, W) = frame.shape[:2]

	if EXPORT_FILE_NAME is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(EXPORT_FILE_NAME, fourcc, 30, (W, H), True)

	status = "Waiting"
	rects = []

	if totalFrames % SKIP_FRAME_SIZE == 0:

		status = "Detecting"
		trackers = []

		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]

			if confidence > DETECT_CONFIDENCE:
				idx = int(detections[0, 0, i, 1])
				
				if idx!= 15:
					continue

				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")

				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				trackers.append(tracker)

	else:

		for tracker in trackers:
			status = "Tracking"

			tracker.update(rgb)
			pos = tracker.get_position()

			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 1)

			rects.append((startX, startY, endX, endY))

	cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

	center_pt1 = (W // 2, H // 2 - 8)
	up_arrowpt = (W // 2, H // 2 - 30)

	center_pt2 = (W // 2, H // 2 + 8)
	down_arrowpt = (W // 2, H // 2 + 30)

	cv2.arrowedLine(frame, center_pt1, up_arrowpt, (0,0,255), 1)
	cv2.arrowedLine(frame, center_pt2, down_arrowpt, (255,0,0), 1)

	objects = ct.update(rects)

	for (objectID, centroid) in objects.items():

		to = trackableObjects.get(objectID, None)

		if to is None:
			to = TrackableObject(objectID, centroid)

		else:

			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			if not to.counted:
				if direction < 0 and centroid[1] < H // 2:
					totalUp += 1
					to.directiontext = "(up)"
					to.counted = True

				elif direction > 0 and centroid[1] > H // 2:
					totalDown += 1
					to.directiontext = "(dw)"
					to.counted = True

		trackableObjects[objectID] = to
		text = "{}".format(objectID) + to.directiontext
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
		#cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 0), 1)
		
	info = [("Total Up Peoples", totalUp), ("Total Down Peoples", totalDown), ("Status", status),]


	strtotaluptext = "{}".format(totalUp)
	strtotaldowntext = "{}".format(totalDown)


	cv2.putText(frame, strtotaluptext, (W//2+10, H//2 - 14),	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, strtotaldowntext, (W//2+10, H//2 + 24),	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


	# for (i, (k, v)) in enumerate(info):
	# 	text = "{}: {}".format(k, v)
	# 	cv2.putText(frame, text, (10, ((i * 17) + 15)),	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

	if writer is not None:
		writer.write(frame)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

	totalFrames += 1
	fps.update()

fps.stop()

print("elapsed time: {:.2f}".format(fps.elapsed()))
print("FPS: {:.2f}".format(fps.fps()))

if writer is not None:
	writer.release()

vs.release()
vs.release()
cv2.destroyAllWindows()