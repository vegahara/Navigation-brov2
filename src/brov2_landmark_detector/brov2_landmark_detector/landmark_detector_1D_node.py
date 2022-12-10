import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib.markers as plt_markers
import matplotlib.text as plt_text
from scipy.signal import find_peaks, peak_prominences, peak_widths
from math import pi, floor, sqrt, tanh, tan, sin
from typing import List
from csaps import csaps

import sys
sys.path.append('utility_functions')
import utility_functions

from rclpy.node import Node

from brov2_interfaces.msg import SonarProcessed
from nav_msgs.msg import Odometry


class Swath:

    def __init__(self):
        self.swath_port = []        # Port side sonar data
        self.swath_stb = []         # Starboard side sonar data

        self.odom = None            # State of the sonar upon swath arrival
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
                        ('landmark_detector_threshold', 1450),
                        ('cubic_spl_smoothing_param', 1e-5),
                        ('max_landmarks_per_scan', 1.0),
                        ('processing_period', 0.0001),
                        ('n_samples', 1000),
                        ('range_sonar', 30)]
        )
                      
        (sonar_data_topic_name, 
        self.landmark_threshold,  
        self.cubic_spl_smoothing_param,
        self.max_landmarks_per_scan,
        processing_period,
        n_samples,
        sonar_range) = \
        self.get_parameters([
            'sonar_data_topic_name', 
            'landmark_detector_threshold',
            'cubic_spl_smoothing_param',
            'max_landmarks_per_scan',
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

        # Plotting buffers
        self.swath_array_buffer = []    
        self.echo_buffer = []           
        self.shadow_buffer = []         
        self.vel_buffer = []         
        self.yaw_buffer = []          
        self.altitude_buffer = []     
        self.n_msg = 0
        self.sonar = SideScanSonar(
            nS = n_samples.value,
            rng = sonar_range.value
        )

        # For figure plotting
        self.plot_1D = False
        self.plot_2D = False
        self.plot_2D_only_scan_lines = True
        self.plot_2D_for_tuning = False

        # Set text settings
        # Set fontsize for all images and plots
        plt.rcParams.update({'font.size': 20})

        if self.plot_1D:
            self.fig = plt.figure() 
            self.axes = self.fig.add_axes([0.07,0.07,0.9,0.9])

        if self.plot_2D:
            self.fig, \
            (self.ax_sonar, self.ax_vel, 
            self.ax_yaw, self.ax_altitude) = plt.subplots(
                1, 4, 
                sharey=True, 
                gridspec_kw={'width_ratios': [3, 1, 1, 1]}
            )
            self.fig.tight_layout()

        if self.plot_2D_only_scan_lines:
            self.swaths = []

            self.fig = plt.figure()
            self.ax_sonar = self.fig.add_subplot(111)
            self.ax_dummy = self.ax_sonar.twinx()
            self.fig.tight_layout()

        if self.plot_2D_for_tuning:
            self.swaths = []
            self.scanlines = []
            self.echo_buffer_1 = []           
            self.shadow_buffer_1 = []
            self.echo_buffer_2 = []           
            self.shadow_buffer_2 = []
            self.echo_buffer_3 = []           
            self.shadow_buffer_3 = []
            self.threshold_1 = 16
            self.threshold_2 = 17
            self.threshold_3 = 18

            self.fig, \
            (self.ax_sonar, self.ax_sonar_threshold_1, 
            self.ax_sonar_threshold_2, 
            self.ax_sonar_threshold_3,
            self.ax_quality_indicator,
            self.ax_speed) = plt.subplots(
                1, 6, 
                sharey=False, 
                gridspec_kw={'width_ratios': [10, 10, 10, 10, 1, 1]} 
            )
            self.fig.tight_layout()

        self.timer = self.create_timer(
            processing_period.value, self.find_landmarks
        )

        self.get_logger().info("Landmark detector node initialized.")

    def sonar_processed_callback(self, sonar_processed_msg):
        
        swath = Swath()

        swath.altitude = sonar_processed_msg.altitude
        swath.odom = sonar_processed_msg.odom

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

        shadow_landmarks = self.extract_landmarks(
            swath, shadow_prop, self.landmark_threshold.value, is_shadows=True
        )
        echo_landmarks = self.extract_landmarks(
            swath, echo_prop, self.landmark_threshold.value, is_shadows=False
        )
        # shadow_landmarks, echo_landmarks = self.filter_landmarks(
        #     shadow_landmarks, echo_landmarks
        # )

        if self.plot_2D or self.plot_2D_only_scan_lines:

            swath_array = []
            swath_array.extend(swath.swath_port)
            swath_array.extend(swath.swath_stb)
            self.swath_array_buffer.append(swath_array)
            self.swaths.append(swath)  
            self.echo_buffer.append(echo_landmarks)
            self.shadow_buffer.append(shadow_landmarks)
            self.vel_buffer.append(swath.odom.twist.twist.linear.x)
            self.altitude_buffer.append(swath.altitude)

            [w,x,y,z] = [
                swath.odom.pose.pose.orientation.w, 
                swath.odom.pose.pose.orientation.x, 
                swath.odom.pose.pose.orientation.y, 
                swath.odom.pose.pose.orientation.z
            ]
            _pitch, yaw = utility_functions.pitch_yaw_from_quaternion(w, x, y, z)
            self.yaw_buffer.append(yaw)

            if (len(self.swath_array_buffer) == 2999):
                self.get_logger().info("Plotting")
                self.plot_landmarks()
            
        if self.plot_1D:
            self.plot_swath_and_landmarks(swath, echo_landmarks, shadow_landmarks, echo_prop_port)
            input('Press any key to continue')

        if self.plot_2D_for_tuning:
            swath_array = []
            swath_array.extend(swath.swath_port)
            swath_array.extend(swath.swath_stb)
            self.swath_array_buffer.append(swath_array)
            self.swaths.append(swath)  

            shadow_landmarks = self.extract_landmarks(
                swath, shadow_prop, self.threshold_1, is_shadows=True
            )
            echo_landmarks = self.extract_landmarks(
                swath, echo_prop, self.threshold_1, is_shadows=False
            )
            shadow_landmarks, echo_landmarks = self.filter_landmarks(
                shadow_landmarks, echo_landmarks
            )
            self.shadow_buffer_1.append(shadow_landmarks)
            self.echo_buffer_1.append(echo_landmarks)

            shadow_landmarks = self.extract_landmarks(
                swath, shadow_prop, self.threshold_2, is_shadows=True
            )
            echo_landmarks = self.extract_landmarks(
                swath, echo_prop, self.threshold_2, is_shadows=False
            )
            shadow_landmarks, echo_landmarks = self.filter_landmarks(
                shadow_landmarks, echo_landmarks
            )
            self.shadow_buffer_2.append(shadow_landmarks)
            self.echo_buffer_2.append(echo_landmarks)

            shadow_landmarks = self.extract_landmarks(
                swath, shadow_prop, self.threshold_3, is_shadows=True
            )
            echo_landmarks = self.extract_landmarks(
                swath, echo_prop, self.threshold_3, is_shadows=False
            )
            shadow_landmarks, echo_landmarks = self.filter_landmarks(
                shadow_landmarks, echo_landmarks
            )
            self.shadow_buffer_3.append(shadow_landmarks)
            self.echo_buffer_3.append(echo_landmarks)

            if (len(self.swath_array_buffer) == 4890):
                self.get_logger().info("Plotting")
                self.plot_for_tuning()
                 
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


    def extract_landmarks(self, swath: Swath, swath_properties, threshold, is_shadows):
        n_bins = len(swath.swath_port) + len(swath.swath_stb)
        landmarks = [0] * n_bins

        swath_array = []
        swath_array.extend(swath.swath_port)
        swath_array.extend(swath.swath_stb)

        lower_index = 0
        higher_index = 0
        
        for (peak, prominence, width) in swath_properties:

            if (2 * width) / prominence < threshold:
                self.shadow_landmarks.append(Landmark(
                    self.get_global_pos(swath, peak), 
                    width, 
                    prominence
                ))
                if is_shadows:
                    for i in range(peak, n_bins + 1):
                        if swath_array[i] >= swath_array[peak] + prominence / 2:
                            if(abs(swath_array[i] - swath_array[peak] + prominence / 2)) < abs(swath_array[i-1] - swath_array[peak] + prominence / 2):
                                higher_index = i
                            else:
                                higher_index = i-1
                            break
                    for i in range(peak, 0, -1):
                        if swath_array[i] >= swath_array[peak] + prominence / 2:
                            if(abs(swath_array[i] - swath_array[peak] + prominence / 2)) < abs(swath_array[i+1] - swath_array[peak] + prominence / 2):
                                lower_index = i
                            else:
                                lower_index = i+1
                            break
                else:
                    for i in range(peak, n_bins + 1):
                        if swath_array[i] <= swath_array[peak] - prominence / 2:
                            if(abs(swath_array[i] - swath_array[peak] + prominence / 2)) < abs(swath_array[i-1] - swath_array[peak] + prominence / 2):
                                higher_index = i
                            else:
                                higher_index = i-1
                            break
                    for i in range(peak, 0, -1):
                        if swath_array[i] <= swath_array[peak] - prominence / 2:
                            if(abs(swath_array[i] - swath_array[peak] + prominence / 2)) < abs(swath_array[i+1] - swath_array[peak] + prominence / 2):
                                lower_index = i
                            else:
                                lower_index = i+1
                            break

                lower_index = max(lower_index,0)
                higher_index = min(higher_index, n_bins-1)

                landmarks[lower_index:higher_index+1] = [1] * (higher_index - lower_index + 1)
                
        return landmarks

    def filter_landmarks(self, shadow_landmarks, echo_landmarks):
        # Not real landmarks if over x % of the swath is "landmark"
        n_bins = 2 * self.sonar.n_samples
        n_landmark_bins = 0
        for i in range(n_bins):
            if (shadow_landmarks[i] == 1) or (echo_landmarks[i] == 1):
                n_landmark_bins += 1
        
        if (n_landmark_bins / n_bins) > self.max_landmarks_per_scan.value:
            shadow_landmarks = [0] * n_bins
            echo_landmarks = [0] * n_bins

        return shadow_landmarks, echo_landmarks


    def invert_swath(self, swath: Swath):
        inverted_swath = Swath()
        inverted_swath.altitude = swath.altitude
        inverted_swath.odom = swath.odom

        max_intensity = max(max(swath.swath_port), max(swath.swath_stb))
        min_intensity = min(min(swath.swath_port), min(swath.swath_stb))
        inverted_swath.swath_port = [
            (max_intensity - min_intensity) - bin for bin in swath.swath_port
        ]
        inverted_swath.swath_stb = [
            (max_intensity - min_intensity) - bin for bin in swath.swath_stb
        ]

        return inverted_swath

    # Not implemented
    def get_global_pos(self, swath: Swath, peak: int):
        return (0, 0, 0)

    def swath_smoothing(self, swath):
        x = np.linspace(0., self.sonar.n_samples - 1, self.sonar.n_samples)
        spl = csaps(x, swath, x, smooth=self.cubic_spl_smoothing_param.value)
       
        return spl   

    def plot_landmarks(self, vmin = 0.6, vmax = 1.5):

        quality_indicators_old, quality_indicators, ground_ranges, \
            distance_traveled, speeds = \
            self.find_swath_indicators(self.swaths)
                 
        self.shadow_buffer = np.asarray(self.shadow_buffer, dtype = np.float64)
        self.echo_buffer = np.asarray(self.echo_buffer, dtype = np.float64)

        str_el = cv.getStructuringElement(cv.MORPH_RECT, (10,10)) 
        self.shadow_buffer = cv.morphologyEx(self.shadow_buffer, cv.MORPH_CLOSE, str_el)
        self.echo_buffer = cv.morphologyEx(self.echo_buffer, cv.MORPH_CLOSE, str_el) 

        str_el = cv.getStructuringElement(cv.MORPH_RECT, (15,15)) 
        self.shadow_buffer = cv.morphologyEx(self.shadow_buffer, cv.MORPH_OPEN, str_el)
        self.echo_buffer = cv.morphologyEx(self.echo_buffer, cv.MORPH_OPEN, str_el) 

        self.shadow_buffer[self.shadow_buffer == 0] = np.nan
        self.echo_buffer[self.echo_buffer == 0] = np.nan

        self.ax_dummy.imshow(self.swath_array_buffer, cmap='copper', vmin = vmin, vmax = vmax)
        self.ax_sonar.imshow(self.swath_array_buffer, cmap='copper', vmin = vmin, vmax = vmax)
        # self.ax_dummy.imshow(self.swath_array_buffer, cmap='copper')
        # self.ax_sonar.imshow(self.swath_array_buffer, cmap='copper')
        self.ax_dummy.imshow(self.shadow_buffer, cmap='spring', vmax = 1)
        self.ax_dummy.imshow(self.echo_buffer, cmap='summer', vmax = 1)
        self.ax_sonar.set(
            xlabel='Across track', 
            ylabel='Along track', 
        )

        # if not self.plot_2D_only_scan_lines:

        #     self.plot_subplot(
        #         self.vel_buffer, self.ax_vel, 
        #         'u (m/s)', 'Surge velocity'
        #     )
        #     self.plot_subplot(
        #         self.yaw_buffer, self.ax_yaw, 
        #         'psi (rad)', 'Yaw angle'
        #     )
        #     self.plot_subplot(
        #         self.altitude_buffer, self.ax_altitude, 
        #         'alt (m)', 'Altitude'
        #     )

        ticks = [0.0, 500.0, 1000.0, 1500.0, 1999.0]
        labels = ['-1000', '-500', '0', '500', '1000']

        self.ax_sonar.set_xticks(ticks)
        self.ax_sonar.set_xticklabels(labels)

        locs = self.ax_sonar.get_yticks()
        labels = []
    
        for i in locs:
            if i in range(len(distance_traveled)):
                labels.append(('%.2f' % distance_traveled[int(i)]) + ' m')
            else:
                labels.append('')

        self.ax_dummy.set_yticklabels(labels)
        self.ax_sonar.margins(0)

        plt.subplots_adjust(left=0.3, bottom=0.1, right=0.64, top=0.95) # Test data
        # plt.subplots_adjust(left=0.37, bottom=0.1, right=0.58, top=0.95) # Training data
        
        plt.pause(10e-5)
        self.fig.canvas.draw()
        input("Press key to continue")

    def plot_subplot(self, data, ax, xlabel, title): 
        ax.plot(
            data[::-1], 
            [i for i in range(len(data)-1 , -1 , -1)]
        )

        asp = (
            np.diff(ax.get_xlim()[::-1])[0] /
            np.diff(ax.get_ylim())[0]
        )
        asp /= np.abs(
            np.diff(self.ax_sonar.get_xlim())[0] / 
            np.diff(self.ax_sonar.get_ylim())[0]
        )
        asp *= 3 # Same as width ratio between figures
        
        ax.set_aspect(asp)
        ax.set(xlabel = xlabel, title = title, )
          

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


    def plot_swath_and_landmarks(self, swath, echo_landmarks, shadow_landmarks, echo_prop_port):

        linewidth = 3

        peak, prominence, width = echo_prop_port[-1]

        print(peak, prominence, width)

        swath_array = []
        swath_array.extend(swath.swath_port)
        swath_array.extend(swath.swath_stb)

        for i in range(len(swath_array)):

            if (echo_landmarks[i] == 0):
                echo_landmarks[i] = np.nan
            else:
                echo_landmarks[i] = swath_array[i]

            if (shadow_landmarks[i] == 0):
                shadow_landmarks[i] = np.nan
            else:
                shadow_landmarks[i] = swath_array[i]

        self.axes.cla()

        self.axes.plot(
            range(-self.sonar.n_samples,self.sonar.n_samples), 
            swath_array, color='black', lw=linewidth
        )
        self.axes.plot(
            range(-self.sonar.n_samples,self.sonar.n_samples), 
            echo_landmarks, color='g', lw=linewidth
        )
        self.axes.plot(
            range(-self.sonar.n_samples,self.sonar.n_samples), 
            shadow_landmarks, color='m', lw = linewidth
        )
        self.axes.plot(
            peak - self.sonar.n_samples - 1, swath_array[peak], 
            marker=plt_markers.CARETDOWN, color='black', lw = linewidth, markersize=20)
        self.axes.vlines(
            x=peak - self.sonar.n_samples - 1, 
            ymax=swath_array[peak],
            ymin=swath_array[peak] - prominence,
            color='black',
            linestyles='dotted',
            lw=3
        )

        xmin = 0
        xmax = 0

        for i in range(peak, self.sonar.n_samples):
            if swath_array[i] <= swath_array[peak] - prominence / 2:
                if(abs(swath_array[i] - swath_array[peak] + prominence / 2)) < abs(swath_array[i-1] - swath_array[peak] + prominence / 2):
                    xmax = i - self.sonar.n_samples - 1
                else:
                    xmax = i - self.sonar.n_samples - 2
                break
        for i in range(peak, 0, -1):
            if swath_array[i] <= swath_array[peak] - prominence / 2:
                if(abs(swath_array[i] - swath_array[peak] + prominence / 2)) < abs(swath_array[i+1] - swath_array[peak] + prominence / 2):
                    xmin = i - self.sonar.n_samples - 1
                else:
                    xmin = i - self.sonar.n_samples
                break

        self.axes.hlines(
            y = swath_array[peak] - prominence / 2,
            xmin=xmin,
            xmax=xmax,
            linestyles='dashed',
            lw=linewidth
        )

        ticks = [-1000.0, -750.0, -500.0, -250.0, 0.0, 250.0, 500.0, 750.0, 999]
        labels = ['-1000', '-750', '-500', '-250', '0', '250', '500', '750', '1000']
        self.axes.set_xticks(ticks)
        self.axes.set_xticklabels(labels)

        self.axes.legend(
            ["Swath", "Detected echoes", "Detected shadows", 'Peak', 'Prominence', 'Width'], 
            loc="upper right"
        )

        self.axes.axvline(x=0, ymin=0, color='black', linestyle='dashdot', lw=linewidth)

        self.axes.margins(x=0.0, y=0.05)

        self.axes.set(
            xlabel='Across track', 
            ylabel='Echo return intesity'
        )

        plt.gca().axis('tight')

        plt.pause(10e-5)
        self.fig.canvas.draw() 

    def plot_for_tuning(self, vmin = 0.6, vmax = 1.5):
        quality_indicators_old, quality_indicators, ground_ranges, \
            distance_traveled, speeds = \
            self.find_swath_indicators(self.swaths)

        quality_im = []
        speed_im = []
        width_speed_and_quality = 200

        for i in range(width_speed_and_quality):
            quality_im.append(quality_indicators)
        for i in range(width_speed_and_quality):
            speed_im.append(speeds)

        quality_im = np.transpose(np.array(quality_im, dtype = np.float64))
        speed_im = np.transpose(np.array(speed_im, dtype = np.float64))

        self.shadow_buffer_1 = np.asarray(self.shadow_buffer_1, dtype = np.float64)
        self.shadow_buffer_2 = np.asarray(self.shadow_buffer_2, dtype = np.float64)
        self.shadow_buffer_3 = np.asarray(self.shadow_buffer_3, dtype = np.float64)
        self.echo_buffer_1 = np.asarray(self.echo_buffer_1, dtype = np.float64)
        self.echo_buffer_2 = np.asarray(self.echo_buffer_2, dtype = np.float64)
        self.echo_buffer_3 = np.asarray(self.echo_buffer_3, dtype = np.float64)

        # str_el = cv.getStructuringElement(cv.MORPH_RECT, (10,10)) 
        # self.shadow_buffer_1 = cv.morphologyEx(self.shadow_buffer_1, cv.MORPH_CLOSE, str_el)
        # self.shadow_buffer_2 = cv.morphologyEx(self.shadow_buffer_2, cv.MORPH_CLOSE, str_el)
        # self.shadow_buffer_3 = cv.morphologyEx(self.shadow_buffer_3, cv.MORPH_CLOSE, str_el)
        # self.echo_buffer_1 = cv.morphologyEx(self.echo_buffer_1, cv.MORPH_CLOSE, str_el) 
        # self.echo_buffer_2 = cv.morphologyEx(self.echo_buffer_2, cv.MORPH_CLOSE, str_el) 
        # self.echo_buffer_3 = cv.morphologyEx(self.echo_buffer_3, cv.MORPH_CLOSE, str_el) 

        # str_el = cv.getStructuringElement(cv.MORPH_RECT, (15,15)) 
        # self.shadow_buffer_1 = cv.morphologyEx(self.shadow_buffer_1, cv.MORPH_OPEN, str_el)
        # self.shadow_buffer_2 = cv.morphologyEx(self.shadow_buffer_2, cv.MORPH_OPEN, str_el)
        # self.shadow_buffer_3 = cv.morphologyEx(self.shadow_buffer_3, cv.MORPH_OPEN, str_el)
        # self.echo_buffer_1 = cv.morphologyEx(self.echo_buffer_1, cv.MORPH_OPEN, str_el) 
        # self.echo_buffer_2 = cv.morphologyEx(self.echo_buffer_2, cv.MORPH_OPEN, str_el) 
        # self.echo_buffer_3 = cv.morphologyEx(self.echo_buffer_3, cv.MORPH_OPEN, str_el) 

        self.shadow_buffer_1[self.shadow_buffer_1 == 0] = np.nan
        self.shadow_buffer_2[self.shadow_buffer_2 == 0] = np.nan
        self.shadow_buffer_3[self.shadow_buffer_3 == 0] = np.nan    
        self.echo_buffer_1[self.echo_buffer_1 == 0] = np.nan
        self.echo_buffer_2[self.echo_buffer_2 == 0] = np.nan
        self.echo_buffer_3[self.echo_buffer_3 == 0] = np.nan

        # self.ax_sonar.imshow(self.swath_array_buffer, cmap='copper', vmin = vmin, vmax = vmax)
        self.ax_sonar.imshow(self.swath_array_buffer, cmap='copper')
        
        # self.ax_sonar_threshold_1.imshow(self.swath_array_buffer, cmap='copper', vmin = vmin, vmax = vmax)
        self.ax_sonar_threshold_1.imshow(self.swath_array_buffer, cmap='copper')

        self.ax_sonar_threshold_1.imshow(self.shadow_buffer_1, cmap='spring', vmax = 1)
        self.ax_sonar_threshold_1.imshow(self.echo_buffer_1, cmap='summer', vmax = 1)
 
        # self.ax_sonar_threshold_2.imshow(self.swath_array_buffer, cmap='copper', vmin = vmin, vmax = vmax)
        self.ax_sonar_threshold_2.imshow(self.swath_array_buffer, cmap='copper')
        self.ax_sonar_threshold_2.imshow(self.shadow_buffer_2, cmap='spring', vmax = 1)
        self.ax_sonar_threshold_2.imshow(self.echo_buffer_2, cmap='summer', vmax = 1)

        # self.ax_sonar_threshold_3.imshow(self.swath_array_buffer, cmap='copper', vmin = vmin, vmax = vmax)
        self.ax_sonar_threshold_3.imshow(self.swath_array_buffer, cmap='copper')
        self.ax_sonar_threshold_3.imshow(self.shadow_buffer_3, cmap='spring', vmax = 1)
        self.ax_sonar_threshold_3.imshow(self.echo_buffer_3, cmap='summer', vmax = 1)

        quality_cmap = plt_colors.LinearSegmentedColormap.from_list(
            "quality_cmap", list(zip([0.0, 0.5, 1.0], ["red","yellow","green"]))
        )

        self.ax_quality_indicator.imshow(quality_im, cmap = quality_cmap, \
            vmin = 0, vmax = 1)
        self.ax_speed.imshow(speed_im, cmap = 'winter', vmin=0.75, vmax=1.25)

        self.ax_sonar_threshold_1.set_yticks([])
        self.ax_sonar_threshold_2.set_yticks([])
        self.ax_sonar_threshold_3.set_yticks([])
        self.ax_quality_indicator.set_yticks([])
        self.ax_speed.set_xticks([])

        # Trick to get last tick on sonar image
        self.ax_quality_indicator.set_xticks([0.0])
        self.ax_quality_indicator.set_xticklabels(['1000'])

        ax_im_lst = [self.ax_sonar, self.ax_sonar_threshold_1,
            self.ax_sonar_threshold_2, self.ax_sonar_threshold_3]

        ticks = [0.0, 500.0, 1000.0, 1500.0, 2000.0]
        labels = ['-1000', '-500', '0', '500', '1000']

        for ax in ax_im_lst:
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)

        locs = self.ax_speed.get_yticks()
        labels = []
    
        for i in locs:
            if i in range(len(distance_traveled)):
                labels.append(('%.2f' % distance_traveled[int(i)]) + ' m')
            else:
                labels.append('')

        self.ax_speed.set_yticklabels(labels)
        self.ax_speed.yaxis.tick_right()

        self.ax_sonar.set(
            ylabel='Along track', 
            title='Sonar image'
        )
        self.ax_sonar_threshold_1.set(
            title='Threshold: ' + str(self.threshold_1)
        )
        self.ax_sonar_threshold_2.set(
            title='Threshold: ' + str(self.threshold_2)
        )
        self.ax_sonar_threshold_3.set( 
            title='Threshold: ' + str(self.threshold_3)
        )
        self.ax_sonar_threshold_1.set_xlabel(
            'Across track', 
            x = 1.0
        )

        self.fig.subplots_adjust(wspace=0)
        self.ax_sonar.margins(0)
        self.ax_sonar_threshold_1.margins(0)
        self.ax_sonar_threshold_2.margins(0)
        self.ax_sonar_threshold_3.margins(0)
        self.ax_quality_indicator.margins(0)
        self.ax_speed.margins(0)

        plt.subplots_adjust(left=0.1, bottom=0, right=0.89, top=1, wspace=0, hspace=0)
        
        plt.pause(10e-5)
        self.fig.canvas.draw()
        input("Press key to continue")

    def find_swath_indicators(self, swaths, k = 4):

        quality_indicators_old = []
        quality_indicators = []
        distance_traveled = []
        speeds = []
        ground_ranges = []

        old_x = 0
        old_y = 0
        old_yaw = 0
        speed = 0
        current_distance = 0

        # Handle first swath
        [w,x,y,z] = [
                swaths[0].odom.pose.pose.orientation.w, 
                swaths[0].odom.pose.pose.orientation.x, 
                swaths[0].odom.pose.pose.orientation.y, 
                swaths[0].odom.pose.pose.orientation.z
        ]
        _pitch, yaw = \
                utility_functions.pitch_yaw_from_quaternion(w, x, y, z)

        old_yaw = yaw
        old_x = swaths[0].odom.pose.pose.position.x
        old_y = swaths[0].odom.pose.pose.position.y

        speed = sqrt(
            swaths[0].odom.twist.twist.linear.x**2 +
            swaths[0].odom.twist.twist.linear.y**2
        )

        ground_range = (swaths[0].altitude / 
            tan(self.sonar.theta - self.sonar.alpha / 2))

        quality_indicators_old.append(1.0)
        quality_indicators.append(1.0)
        distance_traveled.append(current_distance)
        speeds.append(speed)
        ground_ranges.append(ground_range)

        # Find properties for the rest of the swaths
        for swath in swaths[1:]:
            [w,x,y,z] = [
                    swath.odom.pose.pose.orientation.w, 
                    swath.odom.pose.pose.orientation.x, 
                    swath.odom.pose.pose.orientation.y, 
                    swath.odom.pose.pose.orientation.z
            ]
            _pitch, yaw = \
                    utility_functions.pitch_yaw_from_quaternion(w, x, y, z)
            delta_x = swath.odom.pose.pose.position.x - old_x
            delta_y = swath.odom.pose.pose.position.y - old_y

            delta_dist = sqrt(delta_x**2 + delta_y**2)
            delta_yaw = abs(yaw - old_yaw)

            if delta_yaw == 0:
                q = 1
            else:
                l = delta_dist / tan(delta_yaw)

                q = 0.5 * (tanh(k * ((l / ground_range) - 0.5)) + 1)

            ground_range = (swath.altitude / 
                tan(self.sonar.theta - self.sonar.alpha / 2))
            
            quality_indicators.append(q)
            ground_ranges.append(ground_range)

            if delta_yaw == 0:
                q = 1
            else:
                l = delta_dist / tan(delta_yaw)
                r = self.sonar.range

                q = 0.5 * (tanh(k * ((l / r) - 0.5)) + 1)
            
            quality_indicators_old.append(q)

            speed = sqrt(
                swath.odom.twist.twist.linear.x**2 +
                swath.odom.twist.twist.linear.y**2
            )

            old_x = swath.odom.pose.pose.position.x
            old_y = swath.odom.pose.pose.position.y
            old_yaw = yaw
            current_distance += delta_dist

            
            distance_traveled.append(current_distance)
            speeds.append(speed)    

        return quality_indicators_old, quality_indicators, ground_ranges, distance_traveled, speeds
