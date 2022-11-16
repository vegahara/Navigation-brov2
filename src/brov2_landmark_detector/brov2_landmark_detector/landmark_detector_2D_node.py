import numpy as np
import cv2 as cv
from math import pi, sqrt
import matplotlib.pyplot as plt
from csaps import csaps
import copy

import sys
sys.path.append('utility_functions')
import utility_functions

from rclpy.node import Node

from brov2_interfaces.msg import SonarProcessed

class Swath:

    def __init__(self):
        self.swath_port = []        # Port side sonar data
        self.swath_stb = []         # Starboard side sonar data

        self.odom = None            # State of the sonar upon swath arrival
        self.altitude = None        # Altitude of platform upon swath arrival

class SideScanSonar:

    def __init__(self, nS=1000, rng=30, sensor_angle_placement=pi/4, sensor_opening=pi/3):

        self.n_samples = nS                             # Number of samples per active side and ping
        self.range = rng                                # Sonar range in meters
        self.res = (self.range*1.0)/(self.n_samples*2); # Ping resolution [m] across track. Divided by 2 to make wave traveling both back and forth.
        self.theta = sensor_angle_placement
        self.alpha = sensor_opening

class LandmarkDetector2D(Node):

    def __init__(self):
        super().__init__('landmark_detector')

        self.declare_parameters(namespace='',
            parameters=[('sonar_data_topic_name', 'sonar_processed'),
                        ('n_samples', 1000),
                        ('range_sonar', 30),
                        ('scan_lines_per_frame', 5000),
                        ('processing_period', 0.001),
                        ('d_obj_min', 3.0),
                        ('min_height_shadow', 7),
                        ('max_height_shadow', 50),
                        ('min_corr_area', 2),
                        ('bounding_box_fill_limit', 0.3)]
        )
                      
        (sonar_data_topic_name, 
        n_samples,
        sonar_range,
        self.scan_lines_per_frame,
        processing_period,
        self.d_obj_min,
        self.min_height_shadow,
        self.max_height_shadow,
        self.min_corr_area,
        self.bounding_box_fill_limit
        ) = \
        self.get_parameters([
            'sonar_data_topic_name', 
            'n_samples',
            'range_sonar',
            'scan_lines_per_frame',
            'processing_period',
            'd_obj_min',
            'min_height_shadow',
            'max_height_shadow',
            'min_corr_area',
            'bounding_box_fill_limit'
        ])

        self.sonar_processed_subscription = self.create_subscription(
            SonarProcessed, 
            sonar_data_topic_name.value, 
            self.sonar_processed_callback, 
            qos_profile = 10
        )

        self.sonar = SideScanSonar(
            nS = n_samples.value,
            rng = sonar_range.value
        )

        # For figure plotting
        self.plot_figures = True
        if self.plot_figures:
            self.fig, \
            (self.ax_sonar, self.ax_vel, 
            self.ax_yaw, self.ax_altitude) = plt.subplots(
                1, 4, 
                sharey=True, 
                gridspec_kw={'width_ratios': [3, 1, 1, 1]}
            )
            self.fig.tight_layout()

        self.landmarks = None       # Containing all landmarks so far
        self.swath_buffer = []      # Buffer containing swaths to process

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
        buffer_size = len(self.swath_buffer)
        if not (buffer_size%self.scan_lines_per_frame.value == 0 and 
                buffer_size != 0):
            # print('Buffer size: ', buffer_size)
            return

        self.get_logger().info("Finding landmarks in buffer.")

        swaths = self.swath_buffer[-self.scan_lines_per_frame.value:]

        scanlines = []
        altitudes = []

        for swath in swaths:
            scanline = []

            swath.swath_port = self.swath_smoothing(swath.swath_port)
            swath.swath_stb = self.swath_smoothing(swath.swath_stb)

            scanline.extend(swath.swath_port)
            scanline.extend(swath.swath_stb)
            scanlines.append(scanline)
            altitudes.append(swath.altitude)

        sonar_im = np.asarray(scanlines, dtype=np.float64)

        # sonar_im = cv.GaussianBlur(
        #     sonar_im, ksize = (5,5), 
        #     sigmaX = 1, sigmaY = 1, 
        #     borderType = cv.BORDER_REFLECT_101	
        # )

        threshold = np.nanmean(sonar_im) / 2

        threshold = 0.96

        _ret, shadows = \
            cv.threshold(sonar_im, threshold, 1.0, cv.THRESH_BINARY_INV)
        shadows = shadows.astype(np.uint8)

        # str_el = cv.getStructuringElement(cv.MORPH_RECT, (5,5)) 
        # shadows = cv.morphologyEx(shadows, cv.MORPH_CLOSE, str_el) 

        contours, _ = cv.findContours(shadows, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        unfiltered_shadows = copy.deepcopy(shadows)
        mask = np.zeros(shadows.shape[:2], dtype=shadows.dtype)
        
        for cnt in contours:
            area_shadow = cv.contourArea(cnt)
            x,y,w,h = cv.boundingRect(cnt)

            # Find the bin number of the side of the boundingbox 
            # farest away from the AUV, i.e. the end of the shadow (eos)
            if x < self.sonar.n_samples:
                n_eos = self.sonar.n_samples - x
            else:
                n_eos = x + w - self.sonar.n_samples

            echo_length = 2 * self.sonar.range * n_eos / self.sonar.n_samples
            mean_altitude = np.mean(altitudes[y:y+h])

            # Dont add shadow if echo length is smaller then height,
            # its not physically possible
            if echo_length < mean_altitude:
                continue

            # Find the horizontal distance from nadir to detected object
            d_obj = sqrt(echo_length ** 2 - mean_altitude ** 2)

            corr_area = area_shadow * self.d_obj_min.value / d_obj
            area_bounding_box = w * h

            if not (h < self.min_height_shadow.value or 
                    h > self.max_height_shadow.value or
                    corr_area <= self.min_corr_area.value or
                    area_shadow / area_bounding_box < self.bounding_box_fill_limit.value):
   
                cv.drawContours(mask, [cnt], 0, (255), -1)

        shadows = cv.bitwise_and(shadows,shadows, mask = mask)

        self.landmarks = shadows 

        if self.plot_figures:
            self.plot_landmarks(swaths, scanlines, altitudes, shadows, 
                                unfiltered_shadows = unfiltered_shadows)


    def swath_smoothing(self, swath):
        i = 0
        smoothing_swath = []
        if np.isnan(swath[0]):
            i = 0
            while np.isnan(swath[i]):
                i += 1
            smoothing_swath = swath[i:]
        elif np.isnan(swath[-1]):
            i = self.sonar.n_samples - 1
            while np.isnan(swath[i]):
                i -= 1
            smoothing_swath = swath[:i+1]
        else:
            smoothing_swath = swath
        
        x = np.linspace(0., len(smoothing_swath) - 1, len(smoothing_swath))
        spl = csaps(x, smoothing_swath, x, smooth=1e-2)

        if np.isnan(swath[0]):
            swath[i:] = spl
        elif np.isnan(swath[-1]):
            swath[:i+1] = spl
        else:
            swath = spl
       
        return swath

    def plot_landmarks(self, swaths, scanlines, altitudes, shadows, 
                       velocities = None, yaws = None, unfiltered_shadows = None):

        if velocities is None:
            velocities = []
            for swath in swaths:
                velocities.append(swath.odom.twist.twist.linear.x)

        if yaws is None:
            yaws = []
            for swath in swaths:
                [w,x,y,z] = [
                    swath.odom.pose.pose.orientation.w, 
                    swath.odom.pose.pose.orientation.x, 
                    swath.odom.pose.pose.orientation.y, 
                    swath.odom.pose.pose.orientation.z
                ]
                _pitch, yaw = \
                    utility_functions.pitch_yaw_from_quaternion(w, x, y, z)
                yaws.append(yaw)

        shadows = shadows.astype(np.float64)
        shadows[shadows == 0.0] = np.nan

        self.ax_sonar.imshow(scanlines, cmap='copper', vmin = 0.6)

        if unfiltered_shadows is not None:
            unfiltered_shadows = unfiltered_shadows.astype(np.float64)
            unfiltered_shadows[unfiltered_shadows == 0.0] = np.nan
            self.ax_sonar.imshow(unfiltered_shadows, cmap='summer')

        self.ax_sonar.imshow(shadows, cmap='spring')

        self.ax_sonar.set(
            xlabel='Across track', 
            ylabel='Along track', 
            title='Detected landmarks'
        )

        self.plot_subplot(
            velocities, self.ax_vel, 
            'u (m/s)', 'Surge velocity'
        )
        self.plot_subplot(
            yaws, self.ax_yaw, 
            'psi (rad)', 'Yaw angle'
        )
        self.plot_subplot(
            altitudes, self.ax_altitude, 
            'alt (m)', 'Altitude'
        )

        self.fig.subplots_adjust(wspace=0)
        self.ax_sonar.margins(0)
        
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

          