import sys
sys.path.append('utility_functions')
sys.path.append('utility_classes')
from utility_functions import Timestep
from utility_classes import Swath, SideScanSonar, Landmark, Map

from rclpy.node import Node

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy.spatial.transform import Rotation as R
import copy
import pickle

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
                        ('processing_period', 0.001),
                        ('min_shadow_area', 0.1),
                        ('max_shadow_area', 10.0),
                        ('min_shadow_fill_rate', 0.3),
                        ('min_landmark_height', 0.3)]
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
            qos_profile = 10
        )

        self.sonar = SideScanSonar(
            sonar_n_bins.value, sonar_range.value, 
            sonar_transducer_theta.value, sonar_transducer_alpha.value,
            sonar_transducer_beta.value
        )

        self.swath_buffer = []      # Buffer that contains all unprocessed corrected swaths
        self.landmarks = []
        self.processed_swaths = []
        self.map_full = None
        self.fig = plt.figure(figsize=(12, 6))
        self.n_timesteps = 0

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
                
        swaths = self.swath_buffer
        self.processed_swaths.extend(copy.deepcopy(swaths))
        
        self.swath_buffer = self.swath_buffer[self.swaths_per_map.value//2:]
        
        map_origin_x, map_origin_y, n_rows, n_colums = \
            self.find_map_origin_and_size(swaths)
                      
        echo_map, prob_map, observed_swaths_map, range_map = generate_map(
            n_rows, n_colums, self.sonar.n_bins, 
            self.map_resolution.value, map_origin_x, map_origin_y, 
            swaths, self.sonar.range,
            self.swath_ground_range_resolution.value
        )
     
        echo_map = np.asarray(echo_map, dtype=np.float64)

        map = Map(n_rows,n_colums,self.map_resolution.value)
        map.intensity_map = echo_map
        map.origin = [map_origin_x, map_origin_y]

        self.extend_map(map)

        new_landmarks = []

        threshold = 0.95

        _retval, landmark_candidates_filter = \
            cv.threshold(echo_map, threshold, 1.0, cv.THRESH_BINARY_INV)
        landmark_candidates_filter = landmark_candidates_filter.astype(np.uint8)

        str_el = cv.getStructuringElement(cv.MORPH_RECT, (3,3)) 
        landmark_candidates_filter = cv.morphologyEx(landmark_candidates_filter, cv.MORPH_OPEN, str_el)
        landmark_candidates_filter = cv.morphologyEx(landmark_candidates_filter, cv.MORPH_CLOSE, str_el)

        contours, _ = cv.findContours(landmark_candidates_filter, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        filter = np.zeros(landmark_candidates_filter.shape[:2], dtype=landmark_candidates_filter.dtype)

        for cnt in contours:

            area_shadow = cv.contourArea(cnt) * self.map_resolution.value**2
            x,y,w,h = cv.boundingRect(cnt)
            fill_rate = area_shadow / ((max(w,h)**2)*self.map_resolution.value**2)

            if area_shadow < self.min_shadow_area.value or \
               area_shadow > self.max_shadow_area.value or \
               fill_rate < self.min_shadow_fill_rate.value:
                continue

            cv.drawContours(filter, [cnt], 0, (255), -1)

        threshold = 0.98
        #threshold = self.get_otsu_threshold(echo_map, 4096)
        #print(threshold)

        _retval, landmark_candidates = \
            cv.threshold(echo_map, threshold, 1.0, cv.THRESH_BINARY_INV)
        landmark_candidates = landmark_candidates.astype(np.uint8)

        str_el = cv.getStructuringElement(cv.MORPH_RECT, (7,7)) 
        landmark_candidates = cv.morphologyEx(landmark_candidates, cv.MORPH_OPEN, str_el)
        landmark_candidates = cv.morphologyEx(landmark_candidates, cv.MORPH_CLOSE, str_el)

        landmark_candidates_all = landmark_candidates

        contours, _ = cv.findContours(landmark_candidates, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(landmark_candidates.shape[:2], dtype=landmark_candidates.dtype)
        
        for cnt in contours:

            area_shadow = cv.contourArea(cnt) * self.map_resolution.value**2
            x,y,w,h = cv.boundingRect(cnt)
            fill_rate = area_shadow / ((max(w,h)**2)*self.map_resolution.value**2)

            if area_shadow < self.min_shadow_area.value or \
               area_shadow > self.max_shadow_area.value or \
               fill_rate < self.min_shadow_fill_rate.value:
                continue

            min_ground_range = self.sonar.range
            max_ground_range = 0.0
            
            local_landmark_pos = []
            observed_swaths = []

            temp_im = np.zeros(
                landmark_candidates.shape[:2], 
                dtype=landmark_candidates.dtype
            )
            cv.drawContours(temp_im, [cnt], 0, (255), -1)

            filter_out = True

            for col in range(x,x+w):
                for row in range(y,y+h):

                    if not temp_im[row][col]:
                        continue

                    if filter[row][col]:
                        filter_out = False

                    observed_swaths.extend(observed_swaths_map[row][col])

            if filter_out:
                continue

            # Remove all duplicates in observed_swaths
            observed_swaths = list(set(observed_swaths))

            if len(observed_swaths) == 0:
                continue

            observed_swaths.sort()
            center_swath_idx = observed_swaths[int(len(observed_swaths) // 2)]

            for col in range(x,x+w):
                for row in range(y,y+h):
                    if not (center_swath_idx in observed_swaths_map[row][col]):
                        continue

                    cell_range = range_map[row][col]

                    if cell_range < min_ground_range:
                        min_ground_range = cell_range
                        local_landmark_pos = [col,row]
                
                    if cell_range > max_ground_range:
                        max_ground_range = cell_range

            altitude = swaths[center_swath_idx].altitude

            min_slant_range = np.sqrt(min_ground_range**2 + altitude**2)
            max_slant_range = np.sqrt(max_ground_range**2 + altitude**2)

            landmark_height = altitude * (1 - min_slant_range / max_slant_range)

            if landmark_height < self.min_landmark_height.value:
                continue

            x_pose = swaths[center_swath_idx].odom[0]
            y_pose = swaths[center_swath_idx].odom[1]

            actual_ground_range = np.sqrt(min_slant_range**2 - (altitude - landmark_height)**2)
            global_landmark_pos = [
                map_origin_x - self.map_resolution.value * local_landmark_pos[1],
                map_origin_y + self.map_resolution.value * local_landmark_pos[0] 
            ]

            v = [
                global_landmark_pos[0] - x_pose,
                global_landmark_pos[1] - y_pose
            ]
            v = [
                v[0] / np.sqrt(v[0]**2 + v[1]**2),
                v[1] / np.sqrt(v[0]**2 + v[1]**2),
            ]

            # Corrected position
            global_landmark_pos = [
                global_landmark_pos[0] + v[0] * abs(actual_ground_range - min_ground_range),
                global_landmark_pos[1] + v[1] * abs(actual_ground_range - min_ground_range)
            ]

            landmark_pose_transformation = [
                global_landmark_pos[0] - swaths[0].odom[0],
                global_landmark_pos[1] - swaths[0].odom[1]
            ]

            landmark_range = np.sqrt(
                landmark_pose_transformation[0] ** 2 + 
                landmark_pose_transformation[1] ** 2
            )

            landmark_bearing = (np.pi / 2 \
                              - np.arctan2(landmark_pose_transformation[0] , landmark_pose_transformation[1]) \
                              - swaths[0].odom[4]) \
                              % (2 * np.pi)
            
            self.landmarks.append(Landmark(
                global_landmark_pos[0],
                global_landmark_pos[1],
                landmark_range,
                landmark_bearing,
                landmark_height,
                area_shadow,
                fill_rate
            ))

            new_landmarks.append(self.landmarks[-1])

            cv.drawContours(mask, [cnt], 0, (255), -1)

        # Save for offline SLAM

        # timestep = Timestep(swaths[0].odom, new_landmarks)

        # filename = '/home/repo/Navigation-brov2/images/landmark_detection_data/training_100_swaths/pose_and_landmarks_training_data.pickle'

        # with open(filename, "ab") as f:
        #     pickle.dump(timestep, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Plotting
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Bitstream Vera Sans'
        plt.rcParams['font.size'] = 12
        plt.rcParams['image.aspect'] = 'equal'

        landmark_candidates = cv.bitwise_and(
            landmark_candidates,landmark_candidates, mask = mask
        )
        landmark_candidates = landmark_candidates.astype(np.float64)
        landmark_candidates[landmark_candidates == 0.0] = np.nan
        landmark_candidates_all = landmark_candidates_all.astype(np.float64)
        landmark_candidates_all[landmark_candidates_all == 0.0] = np.nan

        if self.fig == None:
            self.fig = plt.figure(figsize=(12, 6))
        else:
            plt.clf()

        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)

        self.ax1.imshow(self.map_full.intensity_map, cmap='copper', vmin=0.6, vmax=1.4)
        self.ax2.imshow(echo_map, cmap='copper', vmin=0.6, vmax=1.4)
        self.ax2.imshow(landmark_candidates_all, cmap='summer')
        self.ax2.imshow(landmark_candidates, cmap='spring')

        for landmark in self.landmarks:
            self.ax1.scatter(
                (landmark.y - self.map_full.origin[1]) / self.map_full.resolution,
                -(landmark.x - self.map_full.origin[0]) / self.map_full.resolution,
                marker='x', c='k'
            )
        texts = []

        for landmark in new_landmarks:
            self.ax2.scatter(
                (landmark.y - map_origin_y) / self.map_resolution.value,
                -(landmark.x - map_origin_x) / self.map_resolution.value,
                marker='x', c='k'
            )
            texts.append(self.ax2.text(
                x=(landmark.y - map_origin_y) / self.map_resolution.value,
                y=-(landmark.x - map_origin_x) / self.map_resolution.value,
                s=f"$h = {landmark.height:.2f} m$\n$A = {landmark.area:.2f} m^2$\n$fr = {landmark.fill_rate:.2f}$"
            ))

        for swath in self.processed_swaths:
            self.ax1.scatter(
                (swath.odom[1] - self.map_full.origin[1]) / self.map_full.resolution, 
                -(swath.odom[0] - self.map_full.origin[0]) / self.map_full.resolution,
                c='k', 
                edgecolor='none',
                marker='.',
            )
        for swath in swaths:
            self.ax2.scatter(
                (swath.odom[1] - map_origin_y) / self.map_resolution.value, 
                -(swath.odom[0] - map_origin_x) / self.map_resolution.value,
                c='k', 
                edgecolor='none',
                marker='.',
            )

        if len(texts) > 0:
            adjust_text(texts)

        x_labels = []
        x_locations = []
        y_labels = []
        y_locations = []

        tick_distanse = 20 # In meters

        x_tick_start = int((map_origin_x % tick_distanse) / self.map_resolution.value)
        y_tick_start = int((tick_distanse - map_origin_y % tick_distanse) / self.map_resolution.value)

        for i in range(x_tick_start, n_rows, int(tick_distanse/self.map_resolution.value)):
            v = map_origin_x - i * self.map_resolution.value
            x_labels.append(('%.2f' % v) + ' m')
            x_locations.append(i)
        for i in range(y_tick_start, n_colums, int(tick_distanse/self.map_resolution.value)):
            v = map_origin_y + i * self.map_resolution.value
            y_labels.append(('%.2f' % v) + ' m')
            y_locations.append(i)

        self.ax1.set_yticks(x_locations)
        self.ax1.set_yticklabels(x_labels)
        self.ax1.set_xticks(y_locations)
        self.ax1.set_xticklabels(y_labels)
        self.ax2.set_yticks(x_locations)
        self.ax2.set_yticklabels(x_labels)
        self.ax2.set_xticks(y_locations)
        self.ax2.set_xticklabels(y_labels)

        plt.draw()
        plt.pause(0.005)
        # self.n_timesteps += 1
        # plt.savefig('/home/repo/Navigation-brov2/images/landmark_detection_data/training_100_swaths/plt_x' + str(self.n_timesteps))
        
        # plt.show()
        # input('Press any key to continue')
