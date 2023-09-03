import sys
sys.path.append('utility_functions')
sys.path.append('utility_classes')
from utility_functions import Timestep
from utility_classes import Swath, SideScanSonar, Landmark, Map

from rclpy.node import Node

import numpy as np
import cv2 as cv
import matplotlib.cm 
import matplotlib.pyplot as plt
from matplotlib import figure
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from adjustText import adjust_text
from scipy.spatial.transform import Rotation as R
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
import copy
# import pickle

import warnings
warnings.filterwarnings("ignore")

from brov2_interfaces.msg import SwathProcessed, SwathArray

from julia.api import Julia

jl = Julia(compiled_modules=False)
jl.eval('import Pkg; Pkg.activate("src/brov2_map/brov2_map/MapGeneration")')
jl.eval('import MapGeneration')
generate_map = jl.eval('MapGeneration.MapGenerationFunctions.generate_map')

class LandmarkDetector(Node):

    def __init__(self) -> None:
        super().__init__('landmark_detector_node')

        self.declare_parameters(
            namespace='',
            parameters=[('processed_swath_topic', 'swath_processed'),
                        ('sonar_n_bins', 1000),
                        ('sonar_range', 30),
                        ('sonar_transducer_theta', (25*np.pi)/180),
                        ('sonar_transducer_alpha', np.pi/3),
                        ('sonar_transducer_beta', (0.5*np.pi)/3),
                        ('swath_ground_range_resolution', 0.03),
                        ('swaths_per_map', 100),
                        ('map_resolution', 0.1),
                        ('processing_period', 0.1),
                        ('min_shadow_area', 0.4),
                        ('max_shadow_area', 10.0),
                        ('min_shadow_fill_rate', 0.15),
                        ('min_landmark_height', 0.15)]
        )
                      
        (processed_swath_topic, 
        sonar_n_bins,
        sonar_range,
        sonar_transducer_theta,
        sonar_transducer_alpha,
        sonar_transducer_beta,
        self.swath_ground_range_resolution,
        self.swaths_per_map,
        self.map_resolution,
        processing_period,
        self.min_shadow_area,
        self.max_shadow_area,
        self.min_shadow_fill_rate,
        self.min_landmark_height
        ) = \
        self.get_parameters([
            'processed_swath_topic', 
            'sonar_n_bins',
            'sonar_range',
            'sonar_transducer_theta',
            'sonar_transducer_alpha',
            'sonar_transducer_beta',
            'swath_ground_range_resolution',
            'swaths_per_map',
            'map_resolution',
            'processing_period',
            'min_shadow_area',
            'max_shadow_area',
            'min_shadow_fill_rate',
            'min_landmark_height'
        ])

        # self.processed_swath_sub = self.create_subscription(
        #     SwathProcessed, 
        #     processed_swath_topic.value, 
        #     self.processed_swath_callback, 
        #     qos_profile = 10
        # )

        self.swath_array_sub = self.create_subscription(
            SwathArray,
            'swath_array',
            self.swath_array_callback,
            qos_profile = 100
        )

        self.sonar = SideScanSonar(
            sonar_n_bins.value, sonar_range.value, 
            sonar_transducer_theta.value, sonar_transducer_alpha.value,
            sonar_transducer_beta.value
        )

        self.swath_buffer = []      # Buffer that contains all unprocessed corrected swaths
        self.landmarks = []
        self.last_timestep_landmarks = []
        self.processed_swaths = []
        self.map_full = None
        self.n_timesteps = 0

        self.fig1 = None
        self.fig2 = None
        self.fig3 = None
        self.fig4 = None
        self.fig5 = None
        self.fig6 = None

        self.timer = self.create_timer(
            processing_period.value, self.landmark_detection
        )

        self.get_logger().info("Landmark detector node initialized.")


    def processed_swath_callback(self, msg):
        r = R.from_quat([
            msg.odom.pose.pose.orientation.x,
            msg.odom.pose.pose.orientation.y,
            msg.odom.pose.pose.orientation.z,
            msg.odom.pose.pose.orientation.w
        ])
        [yaw, pitch, roll] = r.as_euler('ZYX')

        odom = [
            msg.odom.pose.pose.position.x,
            msg.odom.pose.pose.position.y,
            roll,
            pitch,
            yaw
        ]

        swath = Swath(
            header=msg.header,
            data_port=msg.data_port,
            data_stb=msg.data_stb,
            odom=odom,
            altitude=msg.altitude
        )

        # self.swath_buffer.append(swath)


    def swath_array_callback(self, msg):

        for m in msg.swaths:
            r = R.from_quat([
                m.odom.pose.pose.orientation.x,
                m.odom.pose.pose.orientation.y,
                m.odom.pose.pose.orientation.z,
                m.odom.pose.pose.orientation.w
            ])
            [yaw, pitch, roll] = r.as_euler('ZYX')

            odom = [
                m.odom.pose.pose.position.x,
                m.odom.pose.pose.position.y,
                roll,
                pitch,
                yaw,
                m.odom.pose.covariance
            ]

            swath = Swath(
                header=m.header,
                data_port=m.data_port,
                data_stb=m.data_stb,
                odom=odom,
                altitude=m.altitude
            )

            self.swath_buffer.append(swath)


    def find_map_origin_and_size(self, swaths):
        
        min_x = swaths[0].odom[0]
        max_x = swaths[0].odom[0]
        min_y = swaths[0].odom[1]
        max_y = swaths[0].odom[1]

        for swath in swaths:
            min_x = min(min_x, swath.odom[0])
            max_x = max(max_x, swath.odom[0])
            min_y = min(min_y, swath.odom[1])
            max_y = max(max_y, swath.odom[1])

        map_origin_x = np.floor(max_x + self.sonar.range)
        map_origin_y = np.floor(min_y - self.sonar.range)

        n_rows = int(np.ceil(
            (max_x - 
             min_x + 
             2 * self.sonar.range +
             abs(max_x + self.sonar.range - map_origin_x)) / 
             self.map_resolution.value
        ))
        n_colums = int(np.ceil(
            (max_y - 
             min_y + 
             2 * self.sonar.range + 
             abs(min_y - self.sonar.range - map_origin_y)) / 
             self.map_resolution.value
        ))

        return map_origin_x, map_origin_y, n_rows, n_colums
    
    def extend_map(self, map):

        if self.map_full == None:
            self.map_full = map
            return
        
        map_origin_x = max(map.origin[0], self.map_full.origin[0])
        map_origin_y = min(map.origin[1], self.map_full.origin[1])

        n_rows = int(np.ceil(abs(
            map_origin_x - 
            min(
                map.origin[0] - map.n_rows * map.resolution, 
                self.map_full.origin[0] - self.map_full.n_rows * self.map_full.resolution
            )) /
            self.map_full.resolution
        )) 
        n_colums = int(np.ceil(abs(
            map_origin_y - 
            max(
                map.origin[1] + map.n_colums * map.resolution, 
                self.map_full.origin[1] + self.map_full.n_colums * self.map_full.resolution
            )) /
            self.map_full.resolution
        ))

        # Extend existing map
        map_transformation_x = int(
            abs(map_origin_x - self.map_full.origin[0]) / 
            self.map_full.resolution
        )  
        map_transformation_y = int(
            abs(map_origin_y - self.map_full.origin[1])
            / self.map_full.resolution
        ) 

        new_map = Map(n_rows,n_colums,self.map_full.resolution)
        new_map.origin = [map_origin_x, map_origin_y]
        new_map.intensity_map[map_transformation_x:map_transformation_x+self.map_full.n_rows,
                              map_transformation_y:map_transformation_y+self.map_full.n_colums] = \
                              self.map_full.intensity_map
        
        # Add new map
        map_transformation_x = int(
            abs(map_origin_x - map.origin[0]) / 
            map.resolution
        )  
        map_transformation_y = int(
            abs(map_origin_y - map.origin[1])
            / map.resolution
        ) 

        for row in range(map.n_rows):
            for col in range(map.n_colums):
                if not np.isnan(map.intensity_map[row,col]):
                    new_map.intensity_map[map_transformation_x+row,map_transformation_y+col] = \
                        map.intensity_map[row,col]

        self.map_full = new_map

    def get_otsu_threshold(self, im, n_bins=256):

        flat_im = im.flatten()
        flat_im = flat_im[~np.isnan(flat_im)]
        hist, bins = np.histogram(flat_im, n_bins)
        
        total_pixels = im.shape[0] * im.shape[1]
        probs = hist / total_pixels
        
        best_thresh_ind = 0
        best_var = 0

        plt.plot(hist)
        
        for i in range(n_bins):
            # Calculate probabilities and means of the two classes
            bg_prob = np.sum(probs[:i])
            fg_prob = np.sum(probs[i:])
            bg_mean = np.sum(np.arange(i) * probs[:i]) / bg_prob if bg_prob > 0 else 0
            fg_mean = np.sum(np.arange(i, n_bins) * probs[i:]) / fg_prob if fg_prob > 0 else 0
            
            var = bg_prob * fg_prob * (bg_mean - fg_mean)**2
            
            if var > best_var:
                best_thresh_ind = i
                best_var = var

        return bins[best_thresh_ind+1]
    
    def landmark_detection(self):
        
        if len(self.swath_buffer) < self.swaths_per_map.value:
            return

        self.n_timesteps += 1
        print('Current timestep: ', self.n_timesteps)
        
        low_threshold = 0.96
        # low_threshold = 0.96
        low_threshold_structuring_element_size = 3
        high_threshold = 0.98
        # high_threshold = 0.985
        high_threshold_structuring_element_size = 3

        swaths = copy.deepcopy(self.swath_buffer[:self.swaths_per_map.value])
        self.processed_swaths.extend(copy.deepcopy(swaths))

        for s in self.processed_swaths:
            s.data_port = []
            s.data_stb = []

        self.processed_swaths = list(set(self.processed_swaths))
        
        self.swath_buffer = self.swath_buffer[self.swaths_per_map.value//2:]
        
        map_origin_x, map_origin_y, n_rows, n_colums = \
            self.find_map_origin_and_size(swaths)
        
        probability_threshold = 0.1
                      
        intensity_map, range_map, observed_swaths_map = generate_map(
            n_rows, n_colums, self.sonar.n_bins, 
            self.map_resolution.value, map_origin_x, map_origin_y, 
            swaths, self.sonar.range,
            self.swath_ground_range_resolution.value, probability_threshold
        )
     
        intensity_map = np.asarray(intensity_map, dtype=np.float64)

        map = Map(n_rows,n_colums,self.map_resolution.value)
        map.intensity_map = intensity_map
        map.origin = [map_origin_x, map_origin_y]

        self.extend_map(map)

        new_landmarks = []

        # Low threshold filtering

        _retval, landmark_cand_low_thres_im = \
            cv.threshold(intensity_map, low_threshold, 1.0, cv.THRESH_BINARY_INV)
        landmark_cand_low_thres_im = landmark_cand_low_thres_im.astype(np.uint8)

        str_el = cv.getStructuringElement(
            cv.MORPH_RECT, 
            (low_threshold_structuring_element_size,
             low_threshold_structuring_element_size)
        ) 
        landmark_cand_low_thres_im = cv.morphologyEx(landmark_cand_low_thres_im, cv.MORPH_OPEN, str_el)
        landmark_cand_low_thres_im = cv.morphologyEx(landmark_cand_low_thres_im, cv.MORPH_CLOSE, str_el)

        contours, _ = cv.findContours(landmark_cand_low_thres_im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        filter = np.zeros(landmark_cand_low_thres_im.shape[:2], dtype=landmark_cand_low_thres_im.dtype)

        for cnt in contours:

            area_shadow = cv.contourArea(cnt) * self.map_resolution.value**2
            x,y,w,h = cv.boundingRect(cnt)
            fill_rate = area_shadow / ((max(w,h)**2)*self.map_resolution.value**2)

            if area_shadow < self.min_shadow_area.value or \
               area_shadow > self.max_shadow_area.value or \
               fill_rate < self.min_shadow_fill_rate.value:
                continue

            cv.drawContours(filter, [cnt], 0, (255), -1)

        landmark_low_thres_im = cv.bitwise_and(
            landmark_cand_low_thres_im, landmark_cand_low_thres_im, mask = filter
        )

        # High threshold filtering

        _retval, landmark_cand_high_thres_im = \
            cv.threshold(intensity_map, high_threshold, 1.0, cv.THRESH_BINARY_INV)
        landmark_cand_high_thres_im = landmark_cand_high_thres_im.astype(np.uint8)

        str_el = cv.getStructuringElement(
            cv.MORPH_RECT, 
            (high_threshold_structuring_element_size, 
             high_threshold_structuring_element_size)
        ) 
        landmark_cand_high_thres_im = cv.morphologyEx(landmark_cand_high_thres_im, cv.MORPH_OPEN, str_el)
        landmark_cand_high_thres_im = cv.morphologyEx(landmark_cand_high_thres_im, cv.MORPH_CLOSE, str_el)

        contours, _ = cv.findContours(landmark_cand_high_thres_im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        mask_geom_filtering = np.zeros(
            landmark_cand_high_thres_im.shape[:2], 
            dtype=landmark_cand_high_thres_im.dtype
        )
        mask_high_low_thres_filtering = np.zeros(
            landmark_cand_high_thres_im.shape[:2], 
            dtype=landmark_cand_high_thres_im.dtype
        )
        mask_height_filtering = np.zeros(
            landmark_cand_high_thres_im.shape[:2], 
            dtype=landmark_cand_high_thres_im.dtype
        )
        
        for (n_cnt, cnt) in enumerate(contours):

            area_shadow = cv.contourArea(cnt) * self.map_resolution.value**2
            x,y,w,h = cv.boundingRect(cnt)
            fill_rate = area_shadow / ((max(w,h)**2)*self.map_resolution.value**2)

            if area_shadow < self.min_shadow_area.value or \
               area_shadow > self.max_shadow_area.value or \
               fill_rate < self.min_shadow_fill_rate.value:
                continue

            cv.drawContours(mask_geom_filtering, [cnt], 0, (255), -1)

            min_ground_range = self.sonar.range
            max_ground_range = 0.0
            
            local_landmark_pos = []
            observed_swaths = []

            temp_im = np.zeros(
                landmark_cand_high_thres_im.shape[:2], 
                dtype=landmark_cand_high_thres_im.dtype
            )
            cv.drawContours(temp_im, [cnt], 0, (255), -1)

            filter_out = True
            landmark_at_boarder = False

            for col in range(x,x+w):
                for row in range(y,y+h):

                    if not temp_im[row][col]:
                        continue

                    if filter[row][col]:
                        filter_out = False

                    if np.isnan(intensity_map[row + 1][col]) or \
                       np.isnan(intensity_map[row - 1][col]) or \
                       np.isnan(intensity_map[row][col + 1]) or \
                       np.isnan(intensity_map[row][col - 1]):
                        landmark_at_boarder = True

                    observed_swaths.extend(observed_swaths_map[row][col])

            if filter_out or landmark_at_boarder:
                continue

            # Remove all duplicates in observed_swaths
            observed_swaths = list(set(observed_swaths))

            if len(observed_swaths) == 0:
                continue

            cv.drawContours(mask_high_low_thres_filtering, [cnt], 0, (255), -1)

            observed_swaths.sort()
            center_swath_idx = observed_swaths[int(len(observed_swaths) // 2)]

            for col in range(x,x+w):
                for row in range(y,y+h):
                    if (not (center_swath_idx in observed_swaths_map[row][col])) or \
                       (not temp_im[row][col]):
                        continue

                    cell_range = range_map[row][col]

                    if cell_range < min_ground_range:
                        min_ground_range = cell_range
                        local_landmark_pos_boulder = [col,row]
                
                    if cell_range > max_ground_range:
                        max_ground_range = cell_range
                        local_landmark_pos_hole = [col,row]

            altitude = swaths[center_swath_idx].altitude

            # if self.altitude_attitude_correction:
            #     altitude = swath.altitude * np.cos(roll) * np.cos(pitch)
            # else:
            #     altitude = swath.altitude

            roll = swaths[center_swath_idx].odom[2]
            pitch = swaths[center_swath_idx].odom[3]
            yaw = swaths[center_swath_idx].odom[4]

            corr_alt_port = altitude - \
                            self.sonar.z_offset * np.cos(roll) * np.cos(pitch) + \
                            self.sonar.x_offset * np.sin(pitch) + \
                            self.sonar.y_offset * np.sin(roll) * np.cos(pitch)
            corr_alt_stb = altitude - \
                        self.sonar.z_offset * np.cos(roll) * np.cos(pitch) + \
                        self.sonar.x_offset * np.sin(pitch) - \
                        self.sonar.y_offset * np.sin(roll) * np.cos(pitch)

            x_pose = swaths[center_swath_idx].odom[0]
            y_pose = swaths[center_swath_idx].odom[1]

            v = [
                map_origin_x - self.map_resolution.value * local_landmark_pos_boulder[1] - x_pose,
                map_origin_y + self.map_resolution.value * local_landmark_pos_boulder[0] - y_pose
            ]

            theta = (np.arctan2(v[1], v[0]) - yaw) % (2 * np.pi)

            if theta > np.pi:
                altitude = corr_alt_port

            else:
                altitude = corr_alt_stb

            min_slant_range = np.sqrt(min_ground_range**2 + altitude**2)
            max_slant_range = np.sqrt(max_ground_range**2 + altitude**2)

            diff_slant_range = max_slant_range - min_slant_range

            votes_hole = 0
            votes_boulder = 0

            l = int(round(len(observed_swaths)/4))
            h = int(round(len(observed_swaths) - len(observed_swaths)/4))

            # Make number of swaths uneven to be able to classify.
            if (h-l-1) % 2:
                h += 1

            observed_swaths = observed_swaths[l:h]

            for i in observed_swaths:
                swath = swaths[i]
                if theta > np.pi:
                    y_data = swath.data_port[
                        self.sonar.n_bins - int((max_slant_range + 2*diff_slant_range)/self.sonar.slant_resolution):
                        self.sonar.n_bins - int((min_slant_range - 2*diff_slant_range)/self.sonar.slant_resolution)
                    ]
                    y_data = np.flip(y_data)
                else:
                    y_data = swath.data_stb[
                        int((min_slant_range - 2*diff_slant_range)/self.sonar.slant_resolution):
                        int((max_slant_range + 2*diff_slant_range)/self.sonar.slant_resolution)
                    ]

                y_data = [x for x in y_data if not np.isnan(x)]

                y_data = gaussian_filter(y_data, sigma=10.0, mode = 'nearest')

                x_data = np.arange(0, len(y_data))

                p0_neg = [
                    -100.0,
                    len(x_data) / 2,
                    20,
                    1.0
                ]

                p0_pos = [
                    100.0,
                    len(x_data) / 2,
                    20,
                    1.0
                ]

                bounds = (
                    [-np.inf, -(2*diff_slant_range)/self.sonar.slant_resolution + len(x_data) / 2, 2, -np.inf],
                    [np.inf, (2*diff_slant_range)/self.sonar.slant_resolution + len(x_data) / 2, np.inf, np.inf]
                )

                estimate_valid_neg = True
                estimate_valid_pos = True

                try: 
                    popt_neg, pcov_neg, infodict_neg, _mesg_neg, _ier_neg = curve_fit(
                        gaussian_derivative,
                        x_data,
                        y_data,
                        p0=p0_neg,
                        bounds=bounds,
                        full_output = True                    
                    )

                    # print('popt_neg: ', popt_neg)

                    # save_folder = '/home/repo/Navigation-brov2/images/landmark_detection/classification/'
                    # title = 'gaussian_derivative_fitted_x' + str(self.n_timesteps) + '_neg_n_cnt_' + str(n_cnt) + '_swath_' + str(i) 

                    # if theta > np.pi:
                    #     title += '_port'
                    # else:
                    #     title += '_stb'

                    # plt.clf()
                    # plt.plot(x_data, y_data, c='green', label='Filtered swath')
                    # plt.plot(x_data, gaussian_derivative(x_data, *popt_neg), c='purple', linestyle='--', label='Fitted function')
                    # plt.xlabel('Bin \#')
                    # plt.ylabel('Amplitude')
                    # plt.legend()
                    # plt.savefig(save_folder + title.replace(' ', '_') + '.eps', format='eps', dpi=300.0)

                except:
                    print('Not able to estimate with negative initialization')
                    estimate_valid_neg = False

                try:
                    popt_pos, pcov_pos, infodict_pos, _mesg_pos, _ier_pos = curve_fit(
                        gaussian_derivative,
                        x_data,
                        y_data,
                        p0=p0_pos,
                        bounds=bounds,
                        full_output = True                    
                    )

                    # print('popt_pos: ', popt_pos)

                    # save_folder = '/home/repo/Navigation-brov2/images/landmark_detection/classification/'
                    # title = 'gaussian_derivative_fitted_x' + str(self.n_timesteps) + '_pos_n_cnt_' + str(n_cnt) + '_swath_' + str(i)

                    # if theta > np.pi:
                    #     title += '_port'
                    # else:
                    #     title += '_stb'

                    # plt.clf()
                    # plt.plot(x_data, y_data, c='green', label='Filtered swath')
                    # plt.plot(x_data, gaussian_derivative(x_data, *popt_pos), c='purple', linestyle='--', label='Fitted function')
                    # plt.xlabel('Bin \#')
                    # plt.ylabel('Amplitude')
                    # plt.legend()
                    # plt.savefig(save_folder + title.replace(' ', '_') + '.eps', format='eps', dpi=300.0)

                except:
                    print('Not able to estimate with positive initialization')
                    estimate_valid_pos = False

                if (not estimate_valid_neg) and (not estimate_valid_pos):
                    pass
                elif estimate_valid_neg and (not estimate_valid_pos):
                    if popt_neg[0] > 0:
                        votes_hole += 1
                    else:
                        votes_boulder += 1
                elif (not estimate_valid_neg) and estimate_valid_pos:
                    if popt_pos[0] > 0:
                        votes_hole += 1
                    else:
                        votes_boulder += 1
                else:
                    if (popt_neg[0] > 0) and (popt_pos[0] > 0):
                        votes_hole += 1
                    elif (popt_neg[0] < 0) and (popt_pos[0] < 0):
                        votes_boulder += 1
                    else:
                        sum_err_neg = sum(i*i for i in infodict_neg['fvec'])
                        sum_err_pos = sum(i*i for i in infodict_pos['fvec'])

                        if sum_err_neg < sum_err_pos:
                            if popt_neg[0] > 0:
                                votes_hole += 1
                            else:
                                votes_boulder += 1
                        else:
                            if popt_pos[0] > 0:
                                votes_hole += 1
                            else:
                                votes_boulder += 1

            # print('votes_boulder: ', votes_boulder)
            # print('votes_hole: ', votes_hole)

            if votes_boulder > votes_hole:

                landmark_height = altitude * (1 - min_slant_range / max_slant_range)

                # print("Landmark height:", landmark_height)

                if landmark_height < self.min_landmark_height.value:
                    print('Landmark height below threshold. Discarding landmark!')
                    continue

                actual_ground_range = np.sqrt(min_slant_range**2 - (altitude - landmark_height)**2)

                global_landmark_pos = [
                    map_origin_x - self.map_resolution.value * local_landmark_pos_boulder[1],
                    map_origin_y + self.map_resolution.value * local_landmark_pos_boulder[0] 
                ]

                v = [
                    global_landmark_pos[0] - x_pose,
                    global_landmark_pos[1] - y_pose
                ]

                horisontal_y_offset = self.sonar.y_offset * np.cos(roll) + \
                    self.sonar.z_offset * np.sin(roll)

                if theta > np.pi:
                    v[0] -= horisontal_y_offset * np.cos(yaw - np.pi / 2)
                    v[1] -= horisontal_y_offset * np.sin(yaw - np.pi / 2)
                else:
                    v[0] -= horisontal_y_offset * np.cos(yaw + np.pi / 2)
                    v[1] -= horisontal_y_offset * np.sin(yaw + np.pi / 2)

                v = [
                    v[0] / np.sqrt(v[0]**2 + v[1]**2),
                    v[1] / np.sqrt(v[0]**2 + v[1]**2),
                ]

                # Corrected position
                global_landmark_pos = [
                    global_landmark_pos[0] + v[0] * abs(actual_ground_range - min_ground_range),
                    global_landmark_pos[1] + v[1] * abs(actual_ground_range - min_ground_range)
                ]
            
            elif votes_hole > votes_boulder:

                landmark_height = -altitude * (max_slant_range / min_slant_range - 1)

                # print("Landmark height:", landmark_height)

                if abs(landmark_height) < self.min_landmark_height.value:
                    print('Landmark height below threshold. Discarding landmark!')
                    continue

                actual_ground_range = np.sqrt(max_slant_range**2 - (altitude + landmark_height)**2)

                global_landmark_pos = [
                    map_origin_x - self.map_resolution.value * local_landmark_pos_hole[1],
                    map_origin_y + self.map_resolution.value * local_landmark_pos_hole[0] 
                ]

                v = [
                    global_landmark_pos[0] - x_pose,
                    global_landmark_pos[1] - y_pose
                ]

                horisontal_y_offset = self.sonar.y_offset * np.cos(roll) + \
                    self.sonar.z_offset * np.sin(roll)

                if theta > np.pi:
                    v[0] -= horisontal_y_offset * np.cos(yaw - np.pi / 2)
                    v[1] -= horisontal_y_offset * np.sin(yaw - np.pi / 2)
                else:
                    v[0] -= horisontal_y_offset * np.cos(yaw + np.pi / 2)
                    v[1] -= horisontal_y_offset * np.sin(yaw + np.pi / 2)

                v = [
                    v[0] / np.sqrt(v[0]**2 + v[1]**2),
                    v[1] / np.sqrt(v[0]**2 + v[1]**2),
                ]

                # Corrected position
                global_landmark_pos = [
                    global_landmark_pos[0] - v[0] * abs(actual_ground_range - max_ground_range),
                    global_landmark_pos[1] - v[1] * abs(actual_ground_range - max_ground_range)
                ]

            else:
                print('Not able to classify. Discarding landmark!')
                continue

            sigma_b = 5.0 * (self.map_resolution.value / actual_ground_range)
            sigma_r = np.sqrt(self.map_resolution.value**2 + (0.5 * (max_ground_range - min_ground_range))**2)

            landmark_pose_transformation = [
                global_landmark_pos[0] - swaths[-1].odom[0],
                global_landmark_pos[1] - swaths[-1].odom[1]
            ]

            landmark_range = np.sqrt(
                landmark_pose_transformation[0] ** 2 + 
                landmark_pose_transformation[1] ** 2
            )

            landmark_bearing = (np.pi / 2 \
                              - np.arctan2(landmark_pose_transformation[0] , landmark_pose_transformation[1]) \
                              - swaths[-1].odom[4]) \
                              % (2 * np.pi)
            
            # Remove duplicate landmarks
            keep_landmark = True

            for old_landmark in self.last_timestep_landmarks:
                if np.isclose(global_landmark_pos[0], old_landmark.x, atol=3*self.map_resolution.value) and \
                   np.isclose(global_landmark_pos[1], old_landmark.y, atol=3*self.map_resolution.value):
                    keep_landmark = False

            if keep_landmark:    
                new_landmarks.append(Landmark(
                    global_landmark_pos[0],
                    global_landmark_pos[1],
                    landmark_range,
                    sigma_r,
                    landmark_bearing,
                    sigma_b,
                    landmark_height,
                    area_shadow,
                    fill_rate,
                ))
            else:
                # print('Landmark detected in last timestep. Discarding landmark!')
                continue

            cv.drawContours(mask_height_filtering, [cnt], 0, (255), -1)


        landmark_high_thres_im = cv.bitwise_and(
            landmark_cand_high_thres_im,landmark_cand_high_thres_im, mask = mask_geom_filtering
        )

        landmark_no_height_filtered_im = cv.bitwise_and(
            landmark_high_thres_im,landmark_high_thres_im, mask = mask_high_low_thres_filtering
        )

        landmark_im = cv.bitwise_and(
            landmark_no_height_filtered_im,landmark_no_height_filtered_im, mask = mask_height_filtering
        )

        self.landmarks.extend(new_landmarks)
        self.last_timestep_landmarks = new_landmarks
                    
        # Save for offline SLAM

        # timestep = Timestep(swaths[-1].odom, new_landmarks)

        # filename = '/home/repo/Navigation-brov2/images/landmark_detection/pose_and_landmarks_training_data.pickle'

        # with open(filename, "ab") as f:
        #     pickle.dump(timestep, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Plotting
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.sans-serif'] = 'Charter'
        plt.rcParams['font.size'] = 12
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['image.aspect'] = 'equal'

        tick_distance = 20.0
        save_folder = '/home/repo/Navigation-brov2/images/landmark_detection/training_data_paper_figures/'
        # save_folder = None

        landmark_cand_high_thres_im = landmark_cand_high_thres_im.astype(np.float64)
        landmark_cand_low_thres_im = landmark_cand_low_thres_im.astype(np.float64)
        landmark_high_thres_im = landmark_high_thres_im.astype(np.float64)
        landmark_low_thres_im = landmark_low_thres_im.astype(np.float64)
        landmark_no_height_filtered_im = landmark_no_height_filtered_im.astype(np.float64)
        landmark_im = landmark_im.astype(np.float64)

        landmark_cand_high_thres_im[landmark_cand_high_thres_im == 0.0] = np.nan
        landmark_cand_low_thres_im[landmark_cand_low_thres_im == 0.0] = np.nan
        landmark_high_thres_im[landmark_high_thres_im == 0.0] = np.nan
        landmark_low_thres_im[landmark_low_thres_im == 0.0] = np.nan
        landmark_no_height_filtered_im[landmark_no_height_filtered_im == 0.0] = np.nan
        landmark_im[landmark_im == 0.0] = np.nan

        map_origin = [map_origin_x, map_origin_y]
        
        self.plot_landmarks(map.intensity_map, new_landmarks, landmark_cand_high_thres_im, 
                            landmark_cand_low_thres_im, landmark_high_thres_im,
                            landmark_low_thres_im, landmark_no_height_filtered_im, landmark_im,
                            swaths, map_origin, tick_distance, save_folder)
        

    def plot_landmarks(self, map, landmarks, landmark_cand_high_thres_im, 
                       landmark_cand_low_thres_im, landmark_high_thres_im,
                       landmark_low_thres_im, landmark_no_height_filtered_im,
                       landmark_im, swaths, map_origin,
                       tick_distance, save_folder=None):
        
        vmin = 0.6
        vmax = 1.4
        
        cmap_copper = matplotlib.cm.copper
        cmap_copper.set_bad('w', 1.)
        cmap_summer = matplotlib.cm.summer
        cmap_spring = matplotlib.cm.spring

        # self.fig1 = self.plot_map_and_landmarks(
        #     self.fig1, map, cmap_copper,
        #     [landmark_cand_low_thres_im, landmark_low_thres_im],
        #     [cmap_summer, cmap_spring],
        #     [], swaths, 'x' + str(self.n_timesteps) + ' - geometric filtering low threshold',
        #     vmin, vmax, map_origin, tick_distance, save_folder
        # )

        # self.fig6 = self.plot_map_and_landmarks(
        #     self.fig2, map, cmap_copper,
        #     [landmark_cand_high_thres_im],
        #     [cmap_summer],
        #     [], swaths, 'x' + str(self.n_timesteps) + ' - initial landmark candidates',
        #     vmin, vmax, map_origin, tick_distance, save_folder
        # )

        # self.fig2 = self.plot_map_and_landmarks(
        #     self.fig2, map, cmap_copper,
        #     [landmark_cand_high_thres_im, landmark_high_thres_im],
        #     [cmap_summer, cmap_spring],
        #     [], swaths, 'x' + str(self.n_timesteps) + ' - geometric filtering high threshold',
        #     vmin, vmax, map_origin, tick_distance, save_folder
        # )

        # self.fig3 = self.plot_map_and_landmarks(
        #     self.fig3, map, cmap_copper,
        #     [landmark_high_thres_im, landmark_low_thres_im],
        #     [cmap_summer, cmap_spring],
        #     [], swaths, 'x' + str(self.n_timesteps) + ' - low and high threshold filtering',
        #     vmin, vmax, map_origin, tick_distance, save_folder
        # )

        self.fig4 = self.plot_map_and_landmarks(
            self.fig4, map, cmap_copper,
            [landmark_no_height_filtered_im, landmark_im],
            [cmap_summer, cmap_spring],
            [], swaths, 'x' + str(self.n_timesteps) + ' - height filtering',
            vmin, vmax, map_origin, tick_distance, save_folder + 'height_filtering/'
        )

        # self.fig5 = self.plot_map_and_landmarks(
        #     self.fig5, self.map_full.intensity_map, cmap_copper,
        #     [],[], self.landmarks, self.processed_swaths,
        #     'x' + str(self.n_timesteps) + ' - all landmarks',
        #     vmin, vmax, self.map_full.origin, tick_distance, 
        #     save_folder + 'all_landmarks/', False, 'r'
        # )

        # self.fig6 = self.plot_map_and_landmarks(
        #     self.fig6, map, cmap_copper,
        #     [],[],[], swaths,
        #     'x' + str(self.n_timesteps) + ' - bare_map',
        #     vmin, vmax, map_origin, tick_distance, 
        #     save_folder + 'bare_map/', False, '', False
        # )

        # plt.draw()
        # plt.pause(1.0)

    def plot_map_and_landmarks(self, fig, map, map_cmap, map_layer_lst, cmap_lst, 
                               landmarks, swaths, title, vmin, vmax, 
                               map_origin, tick_distanse=20,  save_folder=None, 
                               plot_text=True, landmark_color='k', plot_new_landmarks=True):
        
        if fig == None or save_folder != None:
            fig = figure.Figure()
        else:
            fig.clf()

        landmark_pos_x1 = [-154.36046239747373, 260.2880514088111]
        landmark_pos_x8 = [-148.90485071438684, 259.9493929864564]
        landmark_pos_x26 = [-144.43946362914818, 263.71548955893553]  

        if self.n_timesteps == 1:
            landmark_pos = [
                (self.last_timestep_landmarks[1].y - map_origin[1]) / self.map_resolution.value,
                -(self.last_timestep_landmarks[1].x - map_origin[0]) / self.map_resolution.value
            ]
            # print([self.last_timestep_landmarks[1].x, self.last_timestep_landmarks[1].y])
        elif self.n_timesteps == 8:
            landmark_pos = [
                (self.last_timestep_landmarks[0].y - map_origin[1]) / self.map_resolution.value,
                -(self.last_timestep_landmarks[0].x - map_origin[0]) / self.map_resolution.value
            ]
            # print([self.last_timestep_landmarks[0].x, self.last_timestep_landmarks[0].y])
        elif self.n_timesteps == 26:
            landmark_pos = [
                (self.last_timestep_landmarks[1].y - map_origin[1]) / self.map_resolution.value,
                -(self.last_timestep_landmarks[1].x - map_origin[0]) / self.map_resolution.value
            ]
            # print([self.last_timestep_landmarks[1].x, self.last_timestep_landmarks[1].y])
        else:
            return None

        ax1 = fig.add_subplot(111)

        ax1.imshow(map, cmap=map_cmap, vmin=vmin, vmax=vmax)

        for landmark_layer, cmap in zip(map_layer_lst, cmap_lst):
            ax1.imshow(landmark_layer, cmap)

        n_rows, n_colums = map.shape

        x_labels = []
        x_locations = []
        y_labels = []
        y_locations = []

        x_tick_start = int((map_origin[0] % tick_distanse) / self.map_resolution.value)
        y_tick_start = int((tick_distanse - map_origin[1] % tick_distanse) / self.map_resolution.value)

        for i in range(x_tick_start, n_rows, int(tick_distanse/self.map_resolution.value)):
            v = map_origin[0] - i * self.map_resolution.value
            x_labels.append('$' + ('%.2f' % v) + '$')
            x_locations.append(i)
        for i in range(y_tick_start, n_colums, int(tick_distanse/self.map_resolution.value)):
            v = map_origin[1] + i * self.map_resolution.value
            y_labels.append('$' + ('%.2f' % v) + '$')
            y_locations.append(i)

        ax1.set_yticks(x_locations, x_labels)
        ax1.set_xticks(y_locations, y_labels)
        ax1.set_ylabel('North [m]')
        ax1.set_xlabel('East [m]')

        for swath in swaths:
            ax1.scatter(
                (swath.odom[1] - map_origin[1]) / self.map_resolution.value, 
                -(swath.odom[0] - map_origin[0]) / self.map_resolution.value,
                c='grey', 
                edgecolor='none',
                marker='.',
            )

        texts = []

        ax1.scatter(
                (landmark_pos_x1[1] - map_origin[1]) / self.map_resolution.value,
                -(landmark_pos_x1[0] - map_origin[0]) / self.map_resolution.value,
                marker='x', c='green'
            )
        ax1.scatter(
                (landmark_pos_x8[1] - map_origin[1]) / self.map_resolution.value,
                -(landmark_pos_x8[0] - map_origin[0]) / self.map_resolution.value,
                marker='x', c='green'
            )
        ax1.scatter(
                (landmark_pos_x26[1] - map_origin[1]) / self.map_resolution.value,
                -(landmark_pos_x26[0] - map_origin[0]) / self.map_resolution.value,
                marker='x', c='green'
            )

        for landmark in landmarks:
            ax1.scatter(
                (landmark.y - map_origin[1]) / self.map_resolution.value,
                -(landmark.x - map_origin[0]) / self.map_resolution.value,
                marker='x', c='k'
            )

            if plot_text:
                texts.append(ax1.text(
                    x=(landmark.y - map_origin[1]) / self.map_resolution.value,
                    y=-(landmark.x - map_origin[0]) / self.map_resolution.value,
                    s= r'$h_l = ' +  ('%.2f' % landmark.height) + ' m$ \n' + '$A_l = ' + ('%.2f' % landmark.area) + ' m^2$ \n' + r'$\rho_{bb} = ' + ('%.2f' % landmark.fill_rate) + '$'
                ))
        if plot_new_landmarks:
            for landmark in self.last_timestep_landmarks:
                ax1.scatter(
                    (landmark.y - map_origin[1]) / self.map_resolution.value,
                    -(landmark.x - map_origin[0]) / self.map_resolution.value,
                    marker='x', c=landmark_color
                )

                if plot_text:
                    texts.append(ax1.text(
                        x=(landmark.y - map_origin[1]) / self.map_resolution.value,
                        y=-(landmark.x - map_origin[0]) / self.map_resolution.value,
                        s= r'$h_l = ' +  ('%.2f' % landmark.height) + ' m$ \n' + '$A_l = ' + ('%.2f' % landmark.area) + ' m^2$ \n' + r'$\rho_{bb} = ' + ('%.2f' % landmark.fill_rate) + '$'
                    ))

        if len(texts) > 0:
            adjust_text(texts)

        if self.n_timesteps == 1:
            axins1 = zoomed_inset_axes(ax1, zoom=3.2, loc='lower left')
        else:      
            axins1 = zoomed_inset_axes(ax1, zoom=3.2, loc='lower right')

        axins1.imshow(map, cmap=map_cmap, vmin=vmin, vmax=vmax)

        for landmark_layer, cmap in zip(map_layer_lst, cmap_lst):
            landmark_layer = landmark_layer.astype(np.uint8)
            contours, _ = cv.findContours(landmark_layer, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            mask = np.zeros(landmark_layer.shape[:2], dtype=landmark_layer.dtype)

            for cnt in contours:
                cv.drawContours(mask, [cnt], 0, (255), 1)

            landmark_im = cv.bitwise_and(
                landmark_layer,landmark_layer, mask = mask
            )
            landmark_im = landmark_im.astype(np.float64)            
            landmark_im[landmark_im == 0.0] = np.nan

            axins1.imshow(landmark_im, cmap)

        zoom_size = 4.3 # metre 

        x1 = int(landmark_pos[0] - zoom_size / self.map_resolution.value)
        x2 = int(landmark_pos[0] + zoom_size / self.map_resolution.value)
        y1 = int(landmark_pos[1] - zoom_size / self.map_resolution.value)
        y2 = int(landmark_pos[1] + zoom_size / self.map_resolution.value)

        axins1.set_xlim(x1, x2)
        axins1.set_ylim(y2, y1)
        axins1.tick_params(labelleft=False, labelbottom=False)

        if self.n_timesteps == 1:
            _patch, pp1, pp2 = mark_inset(ax1, axins1, loc1=2, loc2=4)
            pp1.loc1, pp1.loc2 = 2, 3  # inset corner 1 to origin corner 4 (would expect 1)
            pp2.loc1, pp2.loc2 = 4, 1  # inset corner 3 to origin corner 2 (would expect 3)
        elif self.n_timesteps == 8:
            _patch, pp1, pp2 = mark_inset(ax1, axins1, loc1=1, loc2=3)
            pp1.loc1, pp1.loc2 = 1, 4  # inset corner 1 to origin corner 4 (would expect 1)
            pp2.loc1, pp2.loc2 = 3, 2  # inset corner 3 to origin corner 2 (would expect 3)
        elif self.n_timesteps == 26:
            _patch, pp1, pp2 = mark_inset(ax1, axins1, loc1=1, loc2=2)
            pp1.loc1, pp1.loc2 = 1, 4  # inset corner 1 to origin corner 4 (would expect 1)
            pp2.loc1, pp2.loc2 = 2, 3  # inset corner 3 to origin corner 2 (would expect 3)

        if save_folder != None:
            fig.savefig(save_folder + title.replace(' ', '_') + '.eps', format='eps', dpi=300.0)
            return None
            
        return fig


def gaussian_derivative(x, a, b, c, d):

    return a * (1/(np.sqrt(2 * np.pi) * c **3) * (x - b)) * np.exp(-(((x - b))**2)/(2* c ** 2)) + d