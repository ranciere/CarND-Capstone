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

class TLClassifierSite(object):
    def __init__(self):
        self.model = load_mobilenet('light_classification/mobilenet_model.h5')

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
        else:
            res = TrafficLight.UNKNOWN
        return res
