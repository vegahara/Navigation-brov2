import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
from math import pi, floor
from typing import List

from rclpy.node import Node

from brov2_interfaces.msg import SonarProcessed
from geometry_msgs.msg import Pose


class Swath:

    def __init__(self):
        self.swath_port = []        # Port side sonar data
        self.swath_stb = []         # Starboard side sonar data

        self.pose = None            # State of the sonar upon swath arrival
        self.altitude = None        # Altitude of platform upon swath arrival

class Landmark:

    def __init__(self, global_pos, prominence: float, width: float):
        self.global_pos = global_pos                                        # Global position of landmark
        self.prominence = prominence                        
        self.width = width

# Class for containing all sonar parameters
class SideScanSonar:

    def __init__(self, nS=1000, rng=30, sensor_angle_placement=pi/4, sensor_opening=pi/3):

        self.nSamples = nS                              # Number of samples per active side and ping
        self.range = rng                                # Sonar range in meters
        self.res = (self.range*1.0)/(self.nSamples*2);  # Ping resolution [m] across track. Divided by 2 to make wave traveling both back and forth.
        self.theta = sensor_angle_placement
        self.alpha = sensor_opening

    
class LandmarkDetector1D(Node):

    # def __init__(self, sonar: SideScanSonar, threshold: float = 10):
    #     self.shadow_landmarks = []                      # Containing all detected shadow landmarks 
    #     self.echo_landmarks = []                        # Containing all detected echo landmarks   
    #     self.sonar = sonar 
    #     self.landmark_threshold = threshold             # Arbitrary threshold for which landmarks to throw away

    def __init__(self):
        super().__init__('landmark_detector')

        self.declare_parameters(namespace='',
            parameters=[('sonar_data_topic_name', 'sonar_processed'),
                        ('landmark_detector_threshold', 10)]
        )
                      
        sonar_data_topic_name, self.landmark_threshold = self.get_parameters([
            'sonar_data_topic_name', 
            'landmark_detector_threshold'
        ])
                                                                            
        self.sonar_processed_subscription = self.create_subscription(
            SonarProcessed, 
            sonar_data_topic_name.value, 
            self.sonar_processed_callback, 
            qos_profile = 10
        )
        self.sonar_processed_subscription # prevent unused variable warning
        
        # Landmark detection - initialization
        self.shadow_landmarks = []                      # Containing all detected shadow landmarks 
        self.echo_landmarks = []                        # Containing all detected echo landmarks   

        self.get_logger().info("Landmark detector node initialized.")

    def sonar_processed_callback(self, sonar_processed_msg):
        
        swath = Swath()

        swath.altitude = sonar_processed_msg.altitude
        swath.pose = sonar_processed_msg.pose

        data_stb = sonar_processed_msg.data_stb
        swath.swath_stb = [
            int.from_bytes(byte_val, "big") for byte_val in data_stb
        ] # Big endian

        data_port = sonar_processed_msg.data_port
        swath.swath_port = [
            int.from_bytes(byte_val, "big") for byte_val in data_port
        ] # Big endian

        self.find_landmarks(swath)

    def find_landmarks(self, swath: Swath):

        # Find all properties of swath. Make sure swath are flipped right way
        # To find shadows, swath is flipped
        shadow_properties = []
        echo_properties = []

        swath_inverted = self.invert_swath(swath)
        shadow_properties.extend(self.find_swath_properties(np.flip(swath_inverted.swath_port)))
        shadow_properties.extend(self.find_swath_properties(swath_inverted.swath_stb))
        
        echo_properties.extend(self.find_swath_properties(np.flip(swath.swath_port)))
        echo_properties.extend(self.find_swath_properties(swath.swath_stb))

        _shadow_landmarks = self.extract_landmarks(swath, shadow_properties)
        _echo_landmarks = self.extract_landmarks(swath, echo_properties)

        # How to handle landmark detection when we detect both shadows and echoes? Should they be matched up?


    def find_swath_properties(self, swath: List[np.int8]):
        # Make sure that first element of the swath is the first returned echo,
        # e.g. port swaths should be flipped
        
        peaks, _ = find_peaks(swath) 

        # Remove first peaks as it does not correspond to any landmark
        peaks = np.delete(peaks, 0)

        prominences, left_bases, right_bases = peak_prominences(swath, peaks)
        widths, _, _, _ = peak_widths(swath, peaks, 0.5, (prominences, left_bases, right_bases))

        swath_properties = []

        for peak, prominence, width in zip(peaks, prominences, widths):
            swath_properties.append((peak, prominence, width))  

        return swath_properties

    def extract_landmarks(self, swath: Swath, swath_properties):
        landmarks = np.zeros((1,len(swath.swath_port) + len(swath.swath_stb)))

        for (peak, width, prominence) in swath_properties:
            if (2 * width) / prominence < self.landmark_threshold.value:
                self.shadow_landmarks.append(Landmark(self.get_global_pos(swath, peak), width, prominence))
                landmarks[peak - floor(width/2):peak + floor(width/2)] = 1

        return landmarks


    def invert_swath(self, swath: Swath):
        inverted_swath = Swath()
        inverted_swath.altitude = swath.altitude
        inverted_swath.pose = swath.pose
        inverted_swath.swath_port = [255 - bin for bin in swath.swath_port]
        inverted_swath.swath_stb = [255 - bin for bin in swath.swath_stb]

        return inverted_swath

    # Not implemented
    def get_global_pos(self, swath: Swath, peak: int):
        return (0, 0, 0)