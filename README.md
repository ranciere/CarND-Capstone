#System Integration
##Overview
This is the final project of the Udacity Self-Driving Car Engineer Nanodegree. In this project, the goal is to program a Real Self-Driving Car by creating ROS nodes to implement core functionality of the vehicle systems. The software pipeline contains traffic light detection and classification, trajectory planning, and control.
###The team

| Name          	| Udacity Account Email    |
|-------------------|--------------------------|
| Attila Kesmarki   | attila.kesmarki@nng.com  |
| Lajos Rancz       | lajos.rancz@nng.com      |
| Reka Kovacs       | reka.kovacs@nng.com      |
| Zoltan Szabo      | zoltan.szabo.hbu@nng.com |

## Installation

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

## Rubric points
- Smoothly follows waypoints in the simulator.
- Moves on with the target top speed if it is possible.
- Stops at traffic lights when needed.
- Publishes throttle, steering, and brake commands.

## Software Architecture

![Software Architecture](system_architecture.png "Software Architecture")

### What did we implement?
- Traffic Light Detector and Classifier
- Waypoint Updater
- DBW Node with Controller

### Project Components

#### Traffic Light Detector and Classifier

Car receives image from the camera, our system can detect and classify a traffic light color.
First part is to detect a traffic light and a second part is to classify a color of the detected light. If the traffic light is not detected then UNKNOWN is returned. 

##### Traffic light detector

At first we introduce a new message as TrafficLightWithState, which contains the waypoint index and the current traffic light color. The processing of related data has become significantly simpler. 

Source: /ros/src/tl_detector/tl_detector.py
The publisher of the new message created in line 49:
~~~
        self.upcoming_traffic_light_pub = rospy.Publisher('/traffic_waypoint2', TrafficLightWithState, queue_size=1)
~~~
and publish the new message line 88-104:
~~~
example:
            tl_msg = TrafficLightWithState()
            tl_msg.wpid = self.last_wp
            tl_msg.state = self.last_state 
            self.upcoming_traffic_light_pub.publish(tl_msg)
~~~

The traffic light detector installs a kind of classifier (please see later) depending on the config parameter "is_site" in line 37-41:
~~~
        self.is_site = self.config['is_site']

        if self.is_site:
            self.light_classifier = TLClassifierSite()
        else:
            self.light_classifier = TLClassifier()
~~~

The creation/initialization of the classifier object takes a long time, so we had to move the code of the subscription to the necessary ROS topics after the creation of the classifier object, to ensure the messages are only handled on the fully initialized object. 


##### Traffic Light Classifier

We made two classifiers with different technologies:
1. with image manipulation using OpenCV
2. with trained neural network

Source for 1: src/tl_detector/light_classification/tl_classifier.py

The first Solution with OpenCV: The first idea was to find circular areas on the image (using the circular Hough transform) and to determine their color. Unfortunately, this solution did not bring quite accurate detection because many other circular areas were found in the images (example: on the hillside) so we had to reject this idea.

The final solution with OpenCV: We switched to HSV color scheme. The necessary colors (red, yellow, green) are represented with numbers on only the H (hue) layer of the HSV color scheme. Consequently, we need to operate less data. We cropped the lower third part of the image because this area does not contain relevant information which causes further data reduction.
~~~
line 25-27:
	img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	img_crop = img_hsv[0:400,0:800] 
~~~
We make masks for the three colors with only the very bright level in line 28-36:
~~~
example:
	maskg = cv2.inRange(img_crop, (41,200,200), (67,255,255))
~~~
and count the pixels in the masked areas in line 38-41:
~~~
example:
	gcnt = cv2.countNonZero(maskg)
~~~
after that we use a threshold to determine which color is detected in line 42-55:
~~~
example:
    if gcnt > 30:
        num_pixels[2] = gcnt
    else:
        num_pixels[2] = 0
~~~
We returne UNKNOW if the number of pixels of the all colors is zero in line 62-63. 
~~~
	if(np.all(num_pixels == 0)):
		color = TrafficLight.UNKNOWN
~~~
Otherwise, the detected color is the one that contains the most pixels in line 60, 64-65.  
~~~
    else:
        sort_col = np.argsort(num_pixels)
        color = self.colors[sort_col[2]]
~~~

Source for 2: src/tl_detector/light_classification/tl_classifier_site.py

At first some words about the model. We use the 'MobileNET' predefined network without the top level to classify colors from camera images. The top level is changed to 7 output class as below which contains the all possible combination of the traffic light colors:
~~~
#~ The output vector:
#~ light_types['RED']          = [1,0,0,0,0,0,0]
#~ light_types['GREEN']        = [0,1,0,0,0,0,0]
#~ light_types['YELLOW']       = [0,0,1,0,0,0,0]
#~ light_types['RED_YELLOW']   = [0,0,0,1,0,0,0]
#~ light_types['RED_GREEN']    = [0,0,0,0,1,0,0]
#~ light_types['GREEN_YELLOW'] = [0,0,0,0,0,1,0]
#~ light_types['NO_LIGHT']     = [0,0,0,0,0,0,1]
~~~
To train the model we separate the all image which comes fom the simulator and stores the filename with the correct colors. We used the 80 percent part to train the model and 20 percent to validate the model. The trained model is saved with it's weights into the 'mobilenet_model.h5' file.

This classifier load the trained model in line 11-20 in it's constructor:
~~~
    def __init__(self):
        self.model = load_mobilenet('light_classification/mobilenet_model.h5')
~~~
During the classification the first step to resize the given image to 224x224 pixel because it is the input size of the mobilenet model then predict the output in line 53-56:
~~~
        img = cv2.resize(image, (224,224))
        pred = self.model.predict(img[None, :, :, :])
~~~
The output contains a vector with seven elements of the probabilities of colors. At first we determine the simple colors if the probability of one of them reaches the 80 percent in line 60-68:
~~~
example:
        if pred[0] > 0.8:
            res = TrafficLight.RED
~~~
We handle the complex cases when more than one light is detected at the same time. The algorithm works taking into account the change between consecutive lights in line 76-97.

#### Waypoint Updater

Waypoint updater performs the following at each current pose update.

Source: /ros/src/waypoint_updater/waypoint_updater.py

- Find closest waypoint in line 71-89:
    - Searching for the closest waypoint is done by constructing a k-d tree of waypoints at the start with x and y coordinates as dimensions used to partition the point space. This makes the search O(log n) in time complexity.
    - Once the closest waypoint is found it is checked if closest is ahead or behind the car
    - The closest index determined by equation for hyperplane through closest coords
- Generate lane in line 91-119:
    - If the car is allowed to go, the waypoints are the same as base waypoints (line 115)
    - If the traffic light is red the car will decelerate, and stop at the stopline (line 114)
    - If the traffic light is yellow and the distance between the stopline and the car is smaller than the distance to stop with maximum deceleration rate at yellow lights (specified in the 'decel_limit_yellow_light' configuration variable) , the car will decelerate and stop. Even if the same traffic light will switch to red suddenly.  (line 104-112)
    - When the car needs to stop, then the waypoints are generated based on the calculated speed with max. deceleration rate (line 125-139)
    

#### DBW Node with Controller

Our DBW Node uses the reference controller objects.
- Throttle is controlled via PID controller.
- Steering is controlled with YAW controller.



