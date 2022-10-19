import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences, peak_widths
from math import pi, floor
from typing import List
from csaps import csaps


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

    def __init__(self):
        super().__init__('landmark_detector')

        self.declare_parameters(namespace='',
            parameters=[('sonar_data_topic_name', 'sonar_processed'),
                        ('landmark_detector_threshold', 10),
                        ('n_samples', 1000),
                        ('cubic_spl_smoothing_param', 1e-6),
                        ('processing_period', 0.0001)]
        )
                      
        (sonar_data_topic_name, 
        self.landmark_threshold, 
        self.n_samples, 
        self.cubic_spl_smoothing_param,
        processing_period) = \
        self.get_parameters([
            'sonar_data_topic_name', 
            'landmark_detector_threshold',
            'n_samples',
            'cubic_spl_smoothing_param',
            'processing_period'
        ])
                                                                            
        self.sonar_processed_subscription = self.create_subscription(
            SonarProcessed, 
            sonar_data_topic_name.value, 
            self.sonar_processed_callback, 
            qos_profile = 10
        )
        self.sonar_processed_subscription # prevent unused variable warning
        
        # Landmark detection - initialization
        self.swath_buffer = []          # Buffer for buffering incoming messages
        self.shadow_landmarks = []      # Containing all detected shadow landmarks 
        self.echo_landmarks = []        # Containing all detected echo landmarks 
        self.swath_array_buffer = []    # Buffer used for plotting results
        self.echo_buffer = []           # Buffer used for plotting results
        self.shadow_buffer = []         # Buffer used for plotting results
        self.n_msg = 0

        self.timer = self.create_timer(
            processing_period.value, self.find_landmarks
        )

        self.get_logger().info("Landmark detector node initialized.")

    def sonar_processed_callback(self, sonar_processed_msg):
        
        swath = Swath()

        swath.altitude = sonar_processed_msg.altitude
        swath.pose = sonar_processed_msg.pose

        swath.swath_stb = sonar_processed_msg.data_stb
        swath.swath_port = sonar_processed_msg.data_port

        self.swath_buffer.append(swath)

    def find_landmarks(self):
        # Don't process if buffer is empty
        if len(self.swath_buffer) == 0:
            return

        swath = self.swath_buffer[0]
        
        swath.swath_port = self.swath_smoothing(swath.swath_port)
        swath.swath_stb = self.swath_smoothing(swath.swath_stb)

        # Find all  shadow properties of swath
        # To find shadows, swath is inverted
        swath_inverted = self.invert_swath(swath)

        # Find port properties and flip result 
        shadow_prop_port = self.find_swath_properties(swath_inverted.swath_port)
        
        shadow_prop_port.reverse()
        shadow_prop_port = [
            (len(swath.swath_port) - pk, pr, w) for (pk, pr, w) in shadow_prop_port
        ]
        
        # Find stb properties and move indices to be able to work with complete swaths
        shadow_prop_stb = self.find_swath_properties(swath_inverted.swath_stb)
        shadow_prop_stb = [
            (len(swath.swath_port) + pk, pr, w) for (pk, pr, w) in shadow_prop_stb
        ]

        shadow_prop = []
        shadow_prop.extend(shadow_prop_port)
        shadow_prop.extend(shadow_prop_stb)

        # Find all echo properties
        # Find port properties and flip result back
        echo_prop_port = self.find_swath_properties(swath.swath_port)
        
        echo_prop_port.reverse()
        echo_prop_port = [
            (len(swath.swath_port) - pk, pr, w) for (pk, pr, w) in echo_prop_port
        ]
        
        # Find stb properties and move indices to be able to work with complete swaths
        echo_prop_stb = self.find_swath_properties(swath.swath_stb)
        echo_prop_stb = [
            (len(swath.swath_port) + pk, pr, w) for (pk, pr, w) in echo_prop_stb
        ]

        echo_prop = []
        echo_prop.extend(echo_prop_port)
        echo_prop.extend(echo_prop_stb)

        shadow_landmarks = self.extract_landmarks(swath, shadow_prop)
        echo_landmarks = self.extract_landmarks(swath, echo_prop)

        self.plot_landmarks(swath, echo_landmarks, shadow_landmarks)

        self.swath_buffer.pop(0)

        # How to handle landmark detection when we detect both shadows and echoes? Should they be matched up?

    def find_swath_properties(self, swath: Swath):
        # Make sure that first element of the swath is the first returned echo,
        # e.g. port swaths should be flipped
        
        peaks, _ = find_peaks(swath) 

        # Remove first peak as it does not correspond to any landmark
        peaks = np.delete(peaks, 0)

        prominences, left_bases, right_bases = peak_prominences(swath, peaks)

        widths, _, _, _ = peak_widths(
            swath, peaks, 0.5, (prominences, left_bases, right_bases)
        )

        swath_properties = []

        for peak, prominence, width in zip(peaks, prominences, widths):
            swath_properties.append((peak, prominence, width))  

        # print('N landmarks:', len(swath_properties))
        return swath_properties


    def extract_landmarks(self, swath: Swath, swath_properties):
        landmarks = [0] * (len(swath.swath_port) + len(swath.swath_stb))
        
        for (peak, width, prominence) in swath_properties:

            # if (2 * width) / prominence < self.landmark_threshold.value:
            if True:
                self.shadow_landmarks.append(Landmark(
                    self.get_global_pos(swath, peak), 
                    width, 
                    prominence
                ))
                landmarks[peak - floor(width/2):peak + floor(width/2)] = \
                    [1] * floor(width)

        return landmarks


    def invert_swath(self, swath: Swath):
        inverted_swath = Swath()
        inverted_swath.altitude = swath.altitude
        inverted_swath.pose = swath.pose
        inverted_swath.swath_port = [
            max(swath.swath_port) - bin for bin in swath.swath_port
        ]
        inverted_swath.swath_stb = [
            max(swath.swath_stb) - bin for bin in swath.swath_stb
        ]

        return inverted_swath

    # Not implemented
    def get_global_pos(self, swath: Swath, peak: int):
        return (0, 0, 0)

    def plot_landmarks(self, swath: Swath, echo_landmarks, shadow_landmarks):
        
        swath_array = []
        swath_array.extend(np.flip(swath.swath_port))
        swath_array.extend(np.flip(swath.swath_stb))

        # print(len(swath.swath_port))
        # print(len(swath.swath_stb))

        # print(len(swath_array))
        print(len(shadow_landmarks))
        print((len(echo_landmarks)))

        # print(swath_array)

        for i in range(len(swath_array)):

            if (echo_landmarks[i] == 0):
                echo_landmarks[i] = np.nan
            else:
                echo_landmarks[i] = echo_landmarks[i] * swath_array[i]
                swath_array[i] = np.nan

            if (shadow_landmarks[i] == 0):
                shadow_landmarks[i] = np.nan
            else:
                shadow_landmarks[i] = shadow_landmarks[i] * swath_array[i]
                swath_array[i] = np.nan  

        self.swath_array_buffer.append(swath_array)
        self.echo_buffer.append(echo_landmarks)
        self.shadow_buffer.append(shadow_landmarks)
        self.n_msg += 1
        print(self.n_msg)   

        if len(self.swath_array_buffer) > 100:
            #plt.imshow(self.echo_buffer, cmap = 'binary')
            plt.imshow(self.swath_array_buffer, cmap = 'YlOrBr')
            plt.show() 
            input("Press key to continue")          

        plt.plot(swath_array, color='k')
        plt.plot(shadow_landmarks, color='r')
        plt.plot(echo_landmarks, color='g')

        plt.show()

        

    def swath_smoothing(self, swath):
        x = np.linspace(0., self.n_samples.value, self.n_samples.value)
        spl = csaps(x, swath, x, smooth=self.cubic_spl_smoothing_param.value)

        return spl            