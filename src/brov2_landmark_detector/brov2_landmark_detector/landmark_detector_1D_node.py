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

        self.n_samples = nS                             # Number of samples per active side and ping
        self.range = rng                                # Sonar range in meters
        self.res = (self.range*1.0)/(self.n_samples*2); # Ping resolution [m] across track. Divided by 2 to make wave traveling both back and forth.
        self.theta = sensor_angle_placement
        self.alpha = sensor_opening


class LandmarkDetector1D(Node):

    def __init__(self):
        super().__init__('landmark_detector')

        self.declare_parameters(namespace='',
            parameters=[('sonar_data_topic_name', 'sonar_processed'),
                        ('landmark_detector_threshold', 200),
                        ('cubic_spl_smoothing_param', 1e-2),
                        ('processing_period', 0.0001),
                        ('n_samples', 1000),
                        ('range_sonar', 90)]
        )
                      
        (sonar_data_topic_name, 
        self.landmark_threshold,  
        self.cubic_spl_smoothing_param,
        processing_period,
        n_samples,
        sonar_range) = \
        self.get_parameters([
            'sonar_data_topic_name', 
            'landmark_detector_threshold',
            'cubic_spl_smoothing_param',
            'processing_period',
            'n_samples',
            'range_sonar'
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
        self.sonar = SideScanSonar(
            nS = n_samples.value,
            rng = sonar_range.value
        )

        # For figure plotting
        self.plot_figures = True
        if self.plot_figures:
            self.fig = plt.figure() 
            self.axes = self.fig.add_axes([0.05,0,0.9,1])

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
        raw_swath = Swath()
        raw_swath.swath_port = swath.swath_port.copy()
        raw_swath.swath_stb = swath.swath_stb.copy()
        
        swath.swath_port = self.swath_smoothing(swath.swath_port)
        swath.swath_stb = self.swath_smoothing(swath.swath_stb)

        # if self.plot_figures:
        #     self.plot_swath(0, raw_swath, swath)
        #     input('Press key to continue')

        # Find all  shadow properties of swath
        # To find shadows, swath is inverted
        swath_inverted = self.invert_swath(swath)

        # Find port properties and flip result 
        shadow_prop_port = \
            self.find_swath_properties(swath_inverted.swath_port[::-1])
        
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
        echo_prop_port = \
            self.find_swath_properties(swath.swath_port[::-1])
        
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

        if self.plot_figures:
            self.plot_landmarks(swath, echo_landmarks, shadow_landmarks)

        self.swath_buffer.pop(0)

        # How to handle landmark detection when we detect both shadows and echoes? Should they be matched up?

    def find_swath_properties(self, swath):
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

        return swath_properties


    def extract_landmarks(self, swath: Swath, swath_properties):
        n_bins = len(swath.swath_port) + len(swath.swath_stb)
        landmarks = [0] * n_bins
        
        for (peak, prominence, width) in swath_properties:

            if (2 * width) / prominence < self.landmark_threshold.value:
                self.shadow_landmarks.append(Landmark(
                    self.get_global_pos(swath, peak), 
                    width, 
                    prominence
                ))

                lower_index = max(peak - floor(width/2), 0)
                higher_index = min(peak + floor(width/2), n_bins)
                landmarks[lower_index:higher_index] = [1] * (higher_index - lower_index)

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

    def swath_smoothing(self, swath):
        x = np.linspace(0., self.sonar.n_samples - 1, self.sonar.n_samples)
        spl = csaps(x, swath, x, smooth=self.cubic_spl_smoothing_param.value)
       
        return spl   

    def plot_landmarks(self, swath: Swath, echo_landmarks, shadow_landmarks):
        
        swath_array = []
        swath_array.extend(swath.swath_port)
        swath_array.extend(swath.swath_stb)

        for i in range(len(swath_array)):

            if (echo_landmarks[i] == 0):
                echo_landmarks[i] = np.nan
            else:
                swath_array[i] = np.nan

            if (shadow_landmarks[i] == 0):
                shadow_landmarks[i] = np.nan
            else:
                swath_array[i] = np.nan

        self.swath_array_buffer.append(swath_array)
        self.echo_buffer.append(echo_landmarks)
        self.shadow_buffer.append(shadow_landmarks)
        self.n_msg += 1
          

        # if len(self.swath_array_buffer) > 500:
                
        #     self.axes.imshow(self.swath_array_buffer, cmap='copper', vmin = 0.6)
        #     self.axes.set(
        #         xlabel='Across track', 
        #         ylabel='Along track', 
        #         title='Detected landmarks'
        #     )
        #     plt.pause(10e-5)
        #     self.fig.canvas.draw()
        #     input("Press key to continue")
          

    def plot_swath(self, ping, raw_swath: Swath, smoothed_swath: Swath):
        self.axes.cla()
        
        self.axes.plot(
            range(-self.sonar.n_samples,0), 
            raw_swath.swath_port, color='blue'
        )
        self.axes.plot(
            range(0,self.sonar.n_samples),  
            raw_swath.swath_stb, color='blue'
        )
        self.axes.plot(
            range(-self.sonar.n_samples,0), 
            smoothed_swath.swath_port,
            range(0,self.sonar.n_samples),  
            smoothed_swath.swath_stb, color='orange', zorder=3
        )
        self.axes.axvline(x=0, ymin=0, color='black', linestyle='dotted')
   
        self.axes.legend(
            ["Across-track signal", "Cubic spline fitted curve"], 
            loc="upper right"
        )
            
        self.axes.set(
            xlabel='# of sample per ping', 
            ylabel='Ping return (log compressed)'
        )
        self.axes.set_title("Ping %i" % ping)
        plt.gca().axis('tight')

        plt.pause(10e-5)
        self.fig.canvas.draw()    

         