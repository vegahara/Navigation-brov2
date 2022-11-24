import numpy as np
import cv2 as cv
from math import pi, sqrt, tanh, sin
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
                        ('scan_lines_per_frame', 16000),
                        ('processing_period', 0.001),
                        ('d_obj_min', 3.0),
                        ('min_height_shadow', 50),
                        ('max_height_shadow', 150),
                        ('min_corr_area', 80),
                        ('bounding_box_fill_limit', 0.3),
                        ('intensity_threshold', 0.9)]
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
        self.bounding_box_fill_limit,
        self.intensity_threshold
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
            'bounding_box_fill_limit',
            'intensity_threshold'
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
        self.plot_figures = False
        if self.plot_figures:
            self.fig, \
            (self.ax_sonar, self.ax_vel, 
            self.ax_yaw, self.ax_altitude) = plt.subplots(
                1, 4, 
                sharey=True, 
                gridspec_kw={'width_ratios': [3, 1, 1, 1]}
            )
            self.fig.tight_layout()

        # Plotting used for tuning 
        self.plot_for_tuning = True
        if self.plot_for_tuning:
            self.fig, \
            (self.ax_sonar, self.ax_sonar_height, 
            self.ax_sonar_area, self.ax_sonar_fill_rate,
            self.ax_sonar_landmarks, self.ax_quality_indicator,
            self.ax_speed) = plt.subplots(
                1, 7, 
                sharey=False, 
                gridspec_kw={'width_ratios': [10, 10, 10, 10, 10, 1, 1]}
            )
            self.fig.tight_layout()

        # Plotting used for tuning of intensity threshold
        self.plot_for_tuning_threshold = False
        if self.plot_for_tuning_threshold:
            self.fig, \
            (self.ax_sonar, self.ax_sonar_threshold_1, 
            self.ax_sonar_threshold_2, 
            self.ax_sonar_threshold_3,
            self.ax_quality_indicator,
            self.ax_speed) = plt.subplots(
                1, 6, 
                sharey=True, 
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

        sonar_im = cv.GaussianBlur(
            sonar_im, ksize = (5,5), 
            sigmaX = 1, sigmaY = 1, 
            borderType = cv.BORDER_REFLECT_101	
        )

        # Take the mean over the 10 smallest elements for improved robustnes
        idx = np.argpartition(sonar_im.flatten(), 100)
        min = np.nanmean(sonar_im.flatten()[idx[:100]])
        mean = np.nanmean(sonar_im)

        print(min)
        print(mean)

        threshold = (mean - min) * self.intensity_threshold.value + min

        print(threshold)

        _ret, shadows = \
            cv.threshold(sonar_im, threshold, 1.0, cv.THRESH_BINARY_INV)
        shadows = shadows.astype(np.uint8)

        str_el = cv.getStructuringElement(cv.MORPH_RECT, (5,5)) 
        shadows = cv.morphologyEx(shadows, cv.MORPH_CLOSE, str_el) 

        contours, _ = cv.findContours(shadows, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        mask = np.zeros(shadows.shape[:2], dtype=shadows.dtype)

        if self.plot_figures:
            shadows_unfiltered = copy.deepcopy(shadows)
        elif self.plot_for_tuning:
            shadows_unfiltered = copy.deepcopy(shadows)
            mask_height_sel = np.zeros(shadows.shape[:2], dtype=shadows.dtype)
            mask_area_sel = np.zeros(shadows.shape[:2], dtype=shadows.dtype)
            mask_fill_sel = np.zeros(shadows.shape[:2], dtype=shadows.dtype)
        elif self.plot_for_tuning_threshold:
            intensity_threshold_1 = 0.60
            intensity_threshold_2 = 0.64
            intensity_threshold_3 = 0.68

            threshold_1 = (mean - min) * intensity_threshold_1 + min
            threshold_2 = (mean - min) * intensity_threshold_2 + min
            threshold_3 = (mean - min) * intensity_threshold_3 + min

            _ret, shadows_1 = \
                cv.threshold(sonar_im, threshold_1, 1.0, cv.THRESH_BINARY_INV)
            shadows_1 = shadows_1.astype(np.uint8)

            _ret, shadows_2 = \
                cv.threshold(sonar_im, threshold_2, 1.0, cv.THRESH_BINARY_INV)
            shadows_2 = shadows_2.astype(np.uint8)

            _ret, shadows_3 = \
                cv.threshold(sonar_im, threshold_3, 1.0, cv.THRESH_BINARY_INV)
            shadows_3 = shadows_3.astype(np.uint8)

        
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

            if self.plot_for_tuning:
                if not (h < self.min_height_shadow.value or 
                        h > self.max_height_shadow.value):
                    cv.drawContours(mask_height_sel, [cnt], 0, (255), -1)
                
                if not (corr_area <= self.min_corr_area.value):
                    cv.drawContours(mask_area_sel, [cnt], 0, (255), -1)

                if not (area_shadow / area_bounding_box < self.bounding_box_fill_limit.value):
                    cv.drawContours(mask_fill_sel, [cnt], 0, (255), -1)


        if self.plot_for_tuning:
            shadows_height_sel = cv.bitwise_and(shadows,shadows, 
                                                mask = mask_height_sel)
            shadows_area_sel = cv.bitwise_and(shadows_height_sel,shadows_height_sel, 
                                                mask = mask_area_sel)
            shadows_fill_sel = cv.bitwise_and(shadows_height_sel,shadows_height_sel, 
                                              mask = mask_fill_sel)

        shadows = cv.bitwise_and(shadows,shadows, mask = mask)

        if self.plot_figures:
            self.plot_landmarks(swaths, scanlines, altitudes, shadows, 
                shadows_unfiltered = shadows_unfiltered)

        elif self.plot_for_tuning:
            self.plot_landmarks_for_tuning(
                swaths, scanlines, shadows, 
                shadows_unfiltered = shadows_unfiltered,
                shadows_height_sel = shadows_height_sel,
                shadows_area_sel = shadows_area_sel,
                shadows_fill_sel = shadows_fill_sel
                )

        elif self.plot_for_tuning_threshold:
            self.plot_landmarks_for_tuning_threshold(
                swaths, scanlines, 
                shadows_1, shadows_2, shadows_3,
                intensity_threshold_1,
                intensity_threshold_2,
                intensity_threshold_3 
            )

        self.landmarks = shadows

    def find_swath_properties(self, swaths, k = 4):

        quality_indicators = []
        distance_traveled = []
        speeds = []

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

        quality_indicators.append(1.0)
        distance_traveled.append(current_distance)
        speeds.append(speed)


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
                l = delta_dist / sin(delta_yaw)
                r = self.sonar.range

                q = 0.5 * (tanh(k * ((l / r) - 0.5)) + 1)

            speed = sqrt(
                swath.odom.twist.twist.linear.x**2 +
                swath.odom.twist.twist.linear.y**2
            )

            old_x = swath.odom.pose.pose.position.x
            old_y = swath.odom.pose.pose.position.y
            old_yaw = yaw
            current_distance += delta_dist

            quality_indicators.append(q)
            distance_traveled.append(current_distance)
            speeds.append(speed)    

        return quality_indicators, distance_traveled, speeds

 
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
                       velocities = None, yaws = None, 
                       shadows_unfiltered = None):

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

        self.ax_sonar.imshow(scanlines, cmap='copper', vmin = 0.6, vmax = 1.5)

        if shadows_unfiltered is not None:
            shadows_unfiltered = shadows_unfiltered.astype(np.float64)
            shadows_unfiltered[shadows_unfiltered == 0.0] = np.nan
            self.ax_sonar.imshow(shadows_unfiltered, cmap='summer')
            
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

    def plot_landmarks_for_tuning(self, swaths, scanlines, shadows, 
                                  shadows_unfiltered, shadows_height_sel,
                                  shadows_area_sel, shadows_fill_sel,
                                  vmin = 0.6, vmax = 1.5):

        quality_indicators, distance_traveled, speeds = \
            self.find_swath_properties(swaths)

        # Invert to get better representation using summer colourmap
        quality_indicators = [(1.0 - x) for x in quality_indicators]

        quality_im = []
        speed_im = []
        width_speed_and_quality = 200

        for i in range(width_speed_and_quality):
            quality_im.append(quality_indicators)
        for i in range(width_speed_and_quality):
            speed_im.append(speeds)

        quality_im = np.transpose(np.array(quality_im, dtype = np.float64))
        speed_im = np.transpose(np.array(speed_im, dtype = np.float64))

        shadows = shadows.astype(np.float64)
        shadows[shadows == 0.0] = np.nan
        shadows_unfiltered = shadows_unfiltered.astype(np.float64)
        shadows_unfiltered[shadows_unfiltered == 0.0] = np.nan
        shadows_height_sel = shadows_height_sel.astype(np.float64)
        shadows_height_sel[shadows_height_sel == 0.0] = np.nan
        shadows_area_sel = shadows_area_sel.astype(np.float64)
        shadows_area_sel[shadows_area_sel == 0.0] = np.nan
        shadows_fill_sel = shadows_fill_sel.astype(np.float64)
        shadows_fill_sel[shadows_fill_sel == 0.0] = np.nan
        
        self.ax_sonar.imshow(scanlines, cmap='copper', vmin = vmin, vmax = vmax)

        self.ax_sonar_height.imshow(scanlines, cmap='copper', vmin = vmin, vmax = vmax)
        self.ax_sonar_height.imshow(shadows_unfiltered, cmap='summer')
        self.ax_sonar_height.imshow(shadows_height_sel, cmap='spring')   

        self.ax_sonar_area.imshow(scanlines, cmap='copper', vmin = vmin, vmax = vmax)
        self.ax_sonar_area.imshow(shadows_height_sel, cmap='summer')
        self.ax_sonar_area.imshow(shadows_area_sel, cmap='spring') 

        self.ax_sonar_fill_rate.imshow(scanlines, cmap='copper', vmin = vmin, vmax = vmax)
        self.ax_sonar_fill_rate.imshow(shadows_area_sel, cmap='summer')
        self.ax_sonar_fill_rate.imshow(shadows_fill_sel, cmap='spring') 

        self.ax_sonar_landmarks.imshow(scanlines, cmap='copper', vmin = vmin, vmax = vmax)
        self.ax_sonar_landmarks.imshow(shadows, cmap='spring')

        self.ax_quality_indicator.imshow(quality_im, cmap = 'summer', vmin = 0, vmax = 1)
        self.ax_speed.imshow(speed_im, cmap = 'winter')

        self.ax_sonar_height.set_yticks([])
        self.ax_sonar_area.set_yticks([])
        self.ax_sonar_fill_rate.set_yticks([])
        self.ax_sonar_landmarks.set_yticks([])
        self.ax_quality_indicator.set_yticks([])
        self.ax_quality_indicator.set_xticks([])
        self.ax_speed.set_xticks([])

        locs = self.ax_speed.get_yticks()
        labels = []
    
        for i in locs:
            if i in range(len(distance_traveled)):
                labels.append(('%.2f' % distance_traveled[int(i)]) + ' m')
            else:
                labels.append('')

        self.ax_speed.set_yticklabels(labels)
        self.ax_speed.yaxis.tick_right()

        self.fig.tight_layout()

        self.ax_sonar.set(
            xlabel='Across track', 
            ylabel='Along track', 
            title='Sonar image'
        )
        self.ax_sonar_height.set(
            xlabel='Across track', 
            title='Landmarks - height filtered'
        )
        self.ax_sonar_area.set(
            xlabel='Across track', 
            title='Landmarks - area filtered'
        )
        self.ax_sonar_fill_rate.set(
            xlabel='Across track', 
            title='Landmarks - fill rate filtered'
        )
        self.ax_sonar_landmarks.set(
            xlabel='Across track', 
            title='Landmarks - result'
        )

        self.fig.subplots_adjust(wspace=0)
        self.ax_sonar.margins(0)
        self.ax_sonar_height.margins(0)
        self.ax_sonar_area.margins(0)
        self.ax_sonar_fill_rate.margins(0)
        self.ax_sonar_landmarks.margins(0)
        self.ax_quality_indicator.margins(0)
        self.ax_speed.margins(0)
        
        plt.pause(10e-5)
        self.fig.canvas.draw()
        input("Press key to continue")

    def plot_landmarks_for_tuning_threshold(
            self, swaths,
            scanlines, shadows_1, shadows_2, shadows_3,
            intensity_threshold_1,
            intensity_threshold_2,
            intensity_threshold_3, 
            vmin = 0.6, vmax = 1.5
        ):

        quality_indicators, distance_traveled, speeds = \
            self.find_swath_properties(swaths)

        shadows_1 = shadows_1.astype(np.float64)
        shadows_1[shadows_1 == 0.0] = np.nan
        shadows_2 = shadows_2.astype(np.float64)
        shadows_2[shadows_2 == 0.0] = np.nan
        shadows_3 = shadows_3.astype(np.float64)
        shadows_3[shadows_3 == 0.0] = np.nan
        
        self.ax_sonar.imshow(scanlines, cmap='copper', vmin = vmin, vmax = vmax)

        self.ax_sonar_threshold_1.imshow(scanlines, cmap='copper', vmin = vmin, vmax = vmax)
        self.ax_sonar_threshold_1.imshow(shadows_1, cmap='summer')
 
        self.ax_sonar_threshold_2.imshow(scanlines, cmap='copper', vmin = vmin, vmax = vmax)
        self.ax_sonar_threshold_2.imshow(shadows_2, cmap='summer')

        self.ax_sonar_threshold_3.imshow(scanlines, cmap='copper', vmin = vmin, vmax = vmax)
        self.ax_sonar_threshold_3.imshow(shadows_3, cmap='summer')

        self.ax_sonar.set(
            xlabel='Across track', 
            ylabel='Along track', 
            title='Sonar image'
        )
        self.ax_sonar_threshold_1.set(
            xlabel='Across track', 
            title='Intensity threshold: ' + str(intensity_threshold_1)
        )
        self.ax_sonar_threshold_2.set(
            xlabel='Across track', 
            title='Intensity threshold: ' + str(intensity_threshold_2)
        )
        self.ax_sonar_threshold_3.set(
            xlabel='Across track', 
            title='Intensity threshold: ' + str(intensity_threshold_3)
        )

        self.fig.subplots_adjust(wspace=0)
        self.ax_sonar.margins(0)
        self.ax_sonar_threshold_1.margins(0)
        self.ax_sonar_threshold_2.margins(0)
        self.ax_sonar_threshold_3.margins(0)
        
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

          