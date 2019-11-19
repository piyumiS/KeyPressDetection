import cv2
import sys
from imutils.video import FPS
import imutils
import time


tracker = cv2.TrackerCSRT_create()


 
# Read video
video = cv2.VideoCapture("D:/Research/trimmed videos/1789-II.mp4")
# "C:\Users\Piyumi\Videos\VID_20191013_102824.mp4"
# video = cv2.VideoCapture("C:/Users/Piyumi/Videos/VID_20191013_102824.mp4")

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

 # Read first frame.
ret, frame = video.read()
if not ret:
    print ('Cannot read video file')
    sys.exit()
   
# Define an initial bounding box
initBB=None
# Calculate Frames per second (FPS)
fps = None

while True:
	ret,frame= video.read()
	frame_no = int(video.get(0))



	if frame is None:

		print("x")
		break
	# frame = imutils.resize(frame, width=600)

	if initBB is not None:
		

		# get the coordinates of the bounding box
		# tracker.update method returns is bool value for tracking is success or not
		# and the coor
		(success,box)=tracker.update(frame)
		print ("frameNo = " +str(frame_no))
		print ("Box = "+str(box))
		

		if success:
			(x,y,w,h)=[int(v)for v in box]
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		#updte frame counter
		fps.update()
		fps.stop()

		# information to display
		info=[
			("Tracker",tracker),
			("Success","Yes"if success else "No"),
			("FPS","{:.2f}".format(fps.fps())),
			("Frame No",frame_no),
		] 
		for (i,(k,v))in enumerate(info):
			text="{}:{}".format(k,v)
			cv2.putText(frame,text,(800,(i*20)+350),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
	# show the output frame
	cv2.imshow("Frame",frame)
	key=cv2.waitKey(1)& 0xFF

	if key== ord("s"):
		initBB=cv2.selectROI("Frame",frame,showCrosshair=True,fromCenter=False)

		tracker.init(frame,initBB)
		fps=FPS().start()

	if key==ord("q"):
		break	
video.release()
cv2.destroyAllWindows()
