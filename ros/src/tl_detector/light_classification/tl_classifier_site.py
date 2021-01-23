from styx_msgs.msg import TrafficLight

import csv
import cv2
import numpy as np
from math import ceil, exp, log
from enum import Enum

from keras.models import load_model

# I had to make TWO totally ugly and disgusting hack because of a Keras bugs: 
# a) https://github.com/keras-team/keras/issues/7431 
# b) https://github.com/keras-team/keras/issues/6462
def load_mobilenet(fname):
    from keras.utils.generic_utils import CustomObjectScope
    import keras.applications as A
    with CustomObjectScope({'relu6': A.mobilenet.relu6,'DepthwiseConv2D': A.mobilenet.DepthwiseConv2D}):
        model = load_model(fname)
        model._make_predict_function()
    return model


#~ The output vector:
#~ light_types['RED']          = [1,0,0,0,0,0,0]
#~ light_types['GREEN']        = [0,1,0,0,0,0,0]
#~ light_types['YELLOW']       = [0,0,1,0,0,0,0]
#~ light_types['RED_YELLOW']   = [0,0,0,1,0,0,0]
#~ light_types['RED_GREEN']    = [0,0,0,0,1,0,0]
#~ light_types['GREEN_YELLOW'] = [0,0,0,0,0,1,0]
#~ light_types['NO_LIGHT']     = [0,0,0,0,0,0,1]

trafficlight_str = {}
trafficlight_str[0] = 'RED'
trafficlight_str[1] = 'YELLOW'
trafficlight_str[2] = 'GREEN'
trafficlight_str[4] = 'UNKNOWN'

class TLClassifierSite(object):
    def __init__(self):
        self.model = load_mobilenet('light_classification/mobilenet_model.h5')
        self.prev_pred = TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Convert to (224, 224)
        img = cv2.resize(image, (224,224))
        # Predict
        pred = self.model.predict(img[None, :, :, :])
        pred = pred[0]
        # Business logic :)
        res = TrafficLight.UNKNOWN
        # Trivial cases
        if pred[0] > 0.8:
            res = TrafficLight.RED
        elif pred[1] > 0.8:
            res = TrafficLight.GREEN
        elif pred[2] > 0.8:
            res = TrafficLight.YELLOW
        elif pred[6] > 0.8:
            res = TrafficLight.UNKNOWN
        # Complex cases
        else:
            res = self.complex_cases(pred)
        self.prev_pred = res
        
        return res

    def complex_cases(self, pred):
        res = TrafficLight.UNKNOWN
        # Based on previous VALID light
        if self.prev_pred == TrafficLight.RED:
            # Only RED -> RED, the RED -> GREEN alternation is possbile, but it is a defensive algorithm
            if pred[0] > 0.5:
                res = TrafficLight.RED
        elif self.prev_pred == TrafficLight.GREEN:
            # GREEN -> GREEN
            if pred[1] > 0.5:
                res = TrafficLight.GREEN
            # GREEN -> YELLOW
            elif pred[2] > 0.5:
                res = TrafficLight.YELLOW
        elif self.prev_pred == TrafficLight.YELLOW:
            # YELLOW -> YELLOW
            if pred[2] > 0.5:
                res = TrafficLight.YELLOW
            # YELLOW -> RED
            elif pred[0] > 0.5:
                res = TrafficLight.RED
        return res
