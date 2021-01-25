import cv2
import numpy as np
from styx_msgs.msg import TrafficLight
#import rospy

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
	self.colors = [TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN]


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
	#in the simulator classify by image processing, only
	#camera image is in RGB color, convert it to HSV to be able to use color info in one plane
	img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	#cut off lower 1/3 of the image where traffic lights cannot be		
	img_crop = img_hsv[0:400,0:800] 
	#create a binary mask for red color. note, that red is at the beginning and at the end of the H circle therefore 2 masks are needed
	mask1r = cv2.inRange(img_crop, (0,50,20), (5,255,255))
	mask2r = cv2.inRange(img_crop, (175,50,20), (180,255,255))
	#combine the 2 masks
	maskr = cv2.bitwise_or(mask1r,mask2r)
	#yellow mask, S (saturation) value is set very high to select 'very yellow' pixels only -there may be much yellow in the background hills with low saturation
	masky = cv2.inRange(img_crop, (26,200,200), (32,255,255))
	#green mask, same applies here as for the yellow mask: high saturation green is selected to be able to tell traffic lights from trees
	maskg = cv2.inRange(img_crop, (41,200,20), (67,255,255))
	
	num_pixels = np.array([0,0,0]) #number of red, yellow, green pixels in this order
	rcnt = cv2.countNonZero(maskr)
	ycnt = cv2.countNonZero(masky)
	gcnt = cv2.countNonZero(maskg)
	if rcnt > 500:
	    num_pixels[0] = rcnt
	else: 
			num_pixels[0] = 0
	
	if ycnt > 500:
	    num_pixels[1] = ycnt
	else: 
			num_pixels[1] = 0
	
	if gcnt > 500:
	    num_pixels[2] = gcnt
	else: 
			num_pixels[2] = 0
	
	#rospy.loginfo("TL_DETECT: {} {} {}".format( str(cv2.countNonZero(maskr)), str(cv2.countNonZero(masky)), str(cv2.countNonZero(maskg)) ) )

	#see which mask yields the highest number of matching pixels -get the index of it in the array
	sort_col = np.argsort(num_pixels)
	
	if(np.all(num_pixels == 0)): #if there are no red, yellow and green pixels in the image then assume we see no traffic lights so set traffic light status to unknown
		color = TrafficLight.UNKNOWN
	else: #if there are non-zero values the color corresponding to the highest value will be returned
		color = self.colors[sort_col[2]]

       
	return color
