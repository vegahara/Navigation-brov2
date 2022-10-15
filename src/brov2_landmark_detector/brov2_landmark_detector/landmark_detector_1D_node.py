import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
from math import pi, floor

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

    def __init__(self, sonar: SideScanSonar, threshold: float = 10):
        self.shadow_landmarks = []                      # Containing all detected shadow landmarks 
        self.echo_landmarks = []                        # Containing all detected echo landmarks   
        self.sonar = sonar 
        self.landmark_threshold = threshold             # Arbitrary threshold for which landmarks to throw away

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

        swath = self.normalize_swath(swath)

        shadow_properties, echo_properties = self.find_swath_properties(swath)

        shadow_landmarks = np.zeros((1,len(swath.swath_port) + len(swath.swath_stb)))
        echo_landmarks = np.zeros((1,len(swath.swath_port) + len(swath.swath_stb)))
        
        for peak, width, prominence in shadow_properties:
            if (2 * width) / prominence < self.landmark_threshold:
                self.shadow_landmarks.append(Landmark(self.get_global_pos(swath, peak)), width, prominence)
                shadow_landmarks[peak - floor(width/2):peak + floor(width/2)] = 1
                
        for peak, width, prominence in echo_properties:
            if (2 * width) / prominence < self.landmark_threshold:
                self.echo_landmarks.append(Landmark(self.get_global_pos(swath, peak)), width, prominence)
                echo_landmarks[peak - floor(width/2):peak + floor(width/2)] = 1

        # How to handle landmark detection when we detect both shadows and echoes? Should they be matched up?

        return shadow_landmarks, echo_landmarks


    def find_swath_properties(self, swath: Swath):

        swath_array = np.flip(swath.swath_port) + swath.swath_stb

        # Find all possible shadows

        # Flip the swath to make all shadows peaks
        swath_flipped = self.flip_swath(swath)

        shadow_peaks_left = find_peaks(swath_flipped.swath_port) 
        shadow_peaks_right = find_peaks(swath_flipped.swath_stb)

        # Remove first peaks as it does not correspond to any landmark and cocatinate shadow peaks
        shadow_peaks_left = np.delete(shadow_peaks_left, 0)
        shadow_peaks_right = np.delete(shadow_peaks_right, 0)
        shadow_peaks = np.flip(shadow_peaks_left) + shadow_peaks_right

        shadow_promineces, shadow_left_bases, shadow_right_bases = peak_prominences(swath_array, shadow_peaks_left)
        shadow_widths = peak_widths(swath_array, shadow_peaks, 0.5, (shadow_promineces, shadow_left_bases, shadow_right_bases))

        # Find all possible echos
        echo_peaks_left = find_peaks(swath.swath_port) 
        
        echo_peaks_right = find_peaks(swath.swath_stb) 

        # Remove first peaks as it does not correspond to any landmark and cocatinate shadow peaks
        echo_peaks_left = np.delete(echo_peaks_left, 0)
        echo_peaks_right = np.delete(echo_peaks_right, 0)
        echo_peaks = np.flip(echo_peaks_left) + echo_peaks_right

        echo_promineces, echo_left_bases, echo_right_bases = peak_prominences(swath_array, echo_peaks_left)
        echo_widths = peak_widths(swath_array, echo_peaks, 0.5, (echo_promineces, echo_left_bases, echo_right_bases))

        shadow_properties = []
        echo_properties = []

        for peak, prominence, width in shadow_peaks, shadow_promineces, shadow_widths:
            shadow_properties.append((peak, prominence, width))  

        for peak, prominence, width in echo_peaks, echo_promineces, echo_widths:
            echo_properties.append((peak, prominence, width)) 

        return shadow_properties, echo_properties


    def flip_swath(self, swath: Swath):
        
        # Do some flipping of the swath
        swath_flipped = swath

        return swath_flipped

    def normalize_swath(self, swath: Swath, min: int = 0, max: int = 1000):
        pass

    def get_global_pos(swath: Swath, peak: int):
        # Calculate global position of landmark
        pass