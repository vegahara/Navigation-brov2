import sys
sys.path.append('utility_functions')
sys.path.append('utility_classes')
import utility_functions
from utility_classes import Swath, SideScanSonar, Landmark

from rclpy.node import Node
from math import pi
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from brov2_interfaces.msg import SwathProcessed

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
                        ('sonar_transducer_theta', pi/4),
                        ('sonar_transducer_alpha', pi/3),
                        ('swath_ground_range_resolution', 0.03),
                        ('swaths_per_map', 1000),
                        ('map_resolution', 0.1),
                        ('processing_period', 0.001),
                        ('min_shadow_area', 0.1),
                        ('min_shadow_fill_rate', 0.1),
                        ('min_landmark_height', 0.1)]
        )
                      
        (processed_swath_topic, 
        sonar_n_bins,
        sonar_range,
        sonar_transducer_theta,
        sonar_transducer_alpha,
        self.swath_ground_range_resolution,
        self.swaths_per_map,
        self.map_resolution,
        processing_period,
        self.min_shadow_area,
        self.min_shadow_fill_rate,
        self.min_landmark_height
        ) = \
        self.get_parameters([
            'processed_swath_topic', 
            'sonar_n_bins',
            'sonar_range',
            'sonar_transducer_theta',
            'sonar_transducer_alpha',
            'swath_ground_range_resolution',
            'swaths_per_map',
            'map_resolution',
            'processing_period',
            'min_shadow_area',
            'min_shadow_fill_rate',
            'min_landmark_height'
        ])

        self.processed_swath_sub = self.create_subscription(
            SwathProcessed, 
            processed_swath_topic.value, 
            self.processed_swath_callback, 
            qos_profile = 10
        )

        self.sonar = SideScanSonar(
            sonar_n_bins.value, sonar_range.value, 
            sonar_transducer_theta.value, sonar_transducer_alpha.value
        )

        self.swath_buffer = []      # Buffer that contains all unprocessed corrected swaths
        self.landmarks = []

        self.timer = self.create_timer(
            processing_period.value, self.landmark_detection
        )

        self.get_logger().info("Landmark detector node initialized.")


    def processed_swath_callback(self, msg):
        pitch, yaw = utility_functions.pitch_yaw_from_quaternion(
            msg.odom.pose.pose.orientation.w, 
            msg.odom.pose.pose.orientation.x, 
            msg.odom.pose.pose.orientation.y, 
            msg.odom.pose.pose.orientation.z
        )

        odom = [
            msg.odom.pose.pose.position.x,
            msg.odom.pose.pose.position.y,
            0, # We dont use roll in map generation
            pitch,
            yaw
        ]

        swath = Swath(
            header=msg.header,
            data_port=np.flip(msg.data_stb),
            data_stb=np.flip(msg.data_port),
            odom=odom,
            altitude=msg.altitude
        )

        self.swath_buffer.append(swath)

    def landmark_detection(self):

        if len(self.swath_buffer) < self.swaths_per_map.value:
            return
        
        swaths = self.swath_buffer[:self.swaths_per_map.value]
        self.swath_buffer = self.swath_buffer[self.swaths_per_map.value:]
        
        map_origin_x, map_origin_y, n_rows, n_colums = \
            self.find_map_origin_and_size(swaths)
        
        echo_map, prob_map, observed_swaths_map, range_map = generate_map(
            n_rows, n_colums, self.sonar.n_bins, 
            self.map_resolution.value, map_origin_x, map_origin_y, 
            swaths, self.sonar.range, 0.5*pi/180, 
            self.swath_ground_range_resolution.value
        )

        echo_map = np.asarray(echo_map, dtype=np.float64)

        threshold = 0.95

        _ret, landmark_candidates = \
            cv.threshold(echo_map, threshold, 1.0, cv.THRESH_BINARY_INV)
        landmark_candidates = landmark_candidates.astype(np.uint8)

        str_el = cv.getStructuringElement(cv.MORPH_RECT, (3,3)) 
        # landmark_candidates = cv.morphologyEx(landmark_candidates, cv.MORPH_CLOSE, str_el)
        # landmark_candidates = cv.morphologyEx(landmark_candidates, cv.MORPH_OPEN, str_el)

        contours, _ = cv.findContours(landmark_candidates, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(landmark_candidates.shape[:2], dtype=landmark_candidates.dtype)
        
        local_landmarks = []

        for cnt in contours:

            area_shadow = cv.contourArea(cnt) * self.map_resolution.value**2
            x,y,w,h = cv.boundingRect(cnt)
            fill_rate = area_shadow / (w*h*self.map_resolution.value**2)

            if area_shadow < self.min_shadow_area.value or \
               fill_rate < self.min_shadow_fill_rate.value:
                continue

            if area_shadow < self.min_shadow_area.value:
                continue

            min_ground_range = self.sonar.range
            max_groud_range = 0.0
            
            local_landmark_pos = []
            observed_swaths = []

            temp_im = np.zeros(
                landmark_candidates.shape[:2], 
                dtype=landmark_candidates.dtype
            )
            cv.drawContours(temp_im, [cnt], 0, (255), -1)

            for col in range(x,x+w):
                for row in range(y,y+h):

                    if not temp_im[row][col]:
                        continue

                    observed_swaths.extend(observed_swaths_map[row][col])

                    cell_range = range_map[row][col]

                    if cell_range < min_ground_range:
                        min_ground_range = cell_range
                        local_landmark_pos = [col,row]
                
                    if cell_range > max_groud_range:
                        max_ground_range = cell_range

            # Remove all duplicates in observed_swaths
            observed_swaths = list(set(observed_swaths))

            if len(observed_swaths) == 0:
                continue

            # landmark_candidates = cv.bitwise_and(
            # landmark_candidates,landmark_candidates, mask = temp_im
            # )
            # landmark_candidates = landmark_candidates.astype(np.float64)
            # landmark_candidates[landmark_candidates == 0.0] = np.nan

            # fig = plt.figure(figsize=(12, 6))

            # ax1 = fig.add_subplot(1, 1, 1)
            # ax1.imshow(echo_map, cmap='copper', vmin=0.6, vmax=1.4)
            # # ax1.imshow(range_map)
            # ax1.imshow(landmark_candidates, cmap='spring')
            # ax1.scatter(x,y,marker='x',c='k')
            # #ax1.scatter(x,y+h,marker='x',c='k')
            # ax1.scatter(x+w,y,marker='x',c='k')
            # #ax1.scatter(x+w,y+h,marker='x',c='k')
            # ax1.scatter(local_landmark_pos[0], local_landmark_pos[1],marker='x')

            # x_labels = []
            # x_locations = []
            # y_labels = []
            # y_locations = []

            # tick_distanse = 20 # In meters

            # for i in range(0, n_rows, int(tick_distanse/self.map_resolution.value)):
            #     v = map_origin_x - i * self.map_resolution.value
            #     x_labels.append(('%.2f' % v) + ' m')
            #     x_locations.append(i)
            # for i in range(0, n_colums, int(tick_distanse/self.map_resolution.value)):
            #     v = map_origin_y + i * self.map_resolution.value
            #     y_labels.append(('%.2f' % v) + ' m')
            #     y_locations.append(i)

            # ax1.set_yticks(x_locations)
            # ax1.set_yticklabels(x_labels)
            # ax1.set_xticks(y_locations)
            # ax1.set_xticklabels(y_labels)

            # plt.draw()
            # plt.pause(0.001)

            # input('Press any key to contnue')

            altitude = 0.0

            for i in observed_swaths:
                altitude += swaths[i].altitude

            altitude /= len(observed_swaths)

            min_slant_range = np.sqrt(min_ground_range**2 + altitude**2)
            max_slant_range = np.sqrt(max_ground_range**2 + altitude**2)

            landmark_height = altitude * (1 - min_slant_range / max_slant_range)

            # print('Height: ', landmark_height)
            # print('Altitude: ', altitude)
            # print('Min_slant_range: ', min_slant_range)
            # print('Max_slant_range: ', max_slant_range)

            if landmark_height < self.min_landmark_height.value:
                continue

            x_avg = 0
            y_avg = 0

            for i in observed_swaths:
                x_avg = swaths[i].odom[0]
                y_avg = swaths[i].odom[1]

            x_avg /= len(observed_swaths)
            y_avg /= len(observed_swaths)

            actual_ground_range = np.sqrt(min_slant_range**2 + (altitude - landmark_height)**2)
            global_landmark_pos = [
                map_origin_x - self.map_resolution.value * local_landmark_pos[0],
                map_origin_y + self.map_resolution.value * local_landmark_pos[1] 
            ]

            v = [
                global_landmark_pos[0] - x_avg,
                global_landmark_pos[1] - y_avg
            ]
            v = [
                v[0] / np.sqrt(v[0]**2 + v[1]**2),
                v[1] / np.sqrt(v[0]**2 + v[1]**2),
            ]
            

            # Corrected position
            # global_landmark_pos = [
            #     global_landmark_pos[0] + v[0] * abs(actual_ground_range - min_ground_range),
            #     global_landmark_pos[1] + v[1] * abs(actual_ground_range - min_ground_range)
            # ]

            self.landmarks.append(Landmark(
                global_landmark_pos[0],
                global_landmark_pos[1],
                landmark_height
            ))

            local_landmarks.append(local_landmark_pos)

            cv.drawContours(mask, [cnt], 0, (255), -1)

        landmark_candidates = cv.bitwise_and(
            landmark_candidates,landmark_candidates, mask = mask
        )
        landmark_candidates = landmark_candidates.astype(np.float64)
        landmark_candidates[landmark_candidates == 0.0] = np.nan

        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(echo_map, cmap='copper', vmin=0.6, vmax=1.4)
        ax1.imshow(landmark_candidates, cmap='spring')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(range_map)
        ax2.imshow(landmark_candidates, cmap='spring')


        for landmark in self.landmarks:
            ax1.scatter(
                -(landmark.x - map_origin_x) / self.map_resolution.value,
                (landmark.y - map_origin_y) / self.map_resolution.value,
                marker='x', c='k'
            )
            ax2.scatter(
                -(landmark.x - map_origin_x) / self.map_resolution.value,
                (landmark.y - map_origin_y) / self.map_resolution.value,
                marker='x', c='k'
            )

        for landmark in local_landmarks:
            ax1.scatter(
                landmark[0], landmark[1], marker='x', c='k'
            )

        for swath in swaths:
            ax1.scatter(
                (swath.odom[1] - map_origin_y) / self.map_resolution.value, 
                -(swath.odom[0] - map_origin_x) / self.map_resolution.value,
                c='k', 
                edgecolor='none',
                marker='.',
            )
            ax2.scatter(
                (swath.odom[1] - map_origin_y) / self.map_resolution.value, 
                -(swath.odom[0] - map_origin_x) / self.map_resolution.value,
                c='k', 
                edgecolor='none',
                marker='.',
            )

        x_labels = []
        x_locations = []
        y_labels = []
        y_locations = []

        tick_distanse = 20 # In meters

        for i in range(0, n_rows, int(tick_distanse/self.map_resolution.value)):
            v = map_origin_x - i * self.map_resolution.value
            x_labels.append(('%.2f' % v) + ' m')
            x_locations.append(i)
        for i in range(0, n_colums, int(tick_distanse/self.map_resolution.value)):
            v = map_origin_y + i * self.map_resolution.value
            y_labels.append(('%.2f' % v) + ' m')
            y_locations.append(i)

        ax1.set_yticks(x_locations)
        ax1.set_yticklabels(x_labels)
        ax1.set_xticks(y_locations)
        ax1.set_xticklabels(y_labels)
        ax2.set_yticks(x_locations)
        ax2.set_yticklabels(x_labels)
        ax2.set_xticks(y_locations)
        ax2.set_xticklabels(y_labels)

        plt.draw()
        plt.pause(0.001)

        input('Press any key to continue')



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

        map_origin_x = max_x + self.sonar.range
        map_origin_y = min_y - self.sonar.range

        n_rows = int(np.ceil(
            (max_x - min_x + 2 * self.sonar.range) / self.map_resolution.value
        ))
        n_colums = int(np.ceil(
            (max_y - min_y + 2 * self.sonar.range) / self.map_resolution.value
        ))

        return map_origin_x, map_origin_y, n_rows, n_colums