import sys
sys.path.append('utility_functions')
import utility_functions

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

from rclpy.node import Node

from brov2_interfaces.msg import SwathProcessed

from julia.api import Julia
jl = Julia(compiled_modules=False)
jl.eval('import Pkg; Pkg.activate("src/brov2_map/brov2_map/MapGeneration")')
jl.eval('import MapGeneration')
generate_map = jl.eval('MapGeneration.MapGenerationFunctions.generate_map')


class Map:
    def __init__(self, n_rows = 100, n_colums = 100, resolution = 0.1, probability_layer = True) -> None:
        self.n_rows = n_rows            # Height of the map in meters
        self.n_colums = n_colums        # Width of the map in meters
        self.resolution = resolution    # Map resolution on meters
        self.origin= None               # The map origin in world coordinates

        # Map consisting of processed intensity returns from the sonar. 
        self.intensity_map = np.zeros(
            (int(n_rows / resolution), int(n_colums/resolution)), 
            dtype=float
        )

        # Map where each cell corresponds to the pobability that the cell has been observed
        if probability_layer:
            self.probability_map = np.zeros(
                (int(n_rows / resolution), int(n_colums/resolution)), 
                dtype=float
            )

class Swath:
    def __init__(self, data_port, data_stb, odom, altitude):
        self.data_port = data_port      # Port side sonar data
        self.data_stb = data_stb        # Starboard side sonar data
        self.odom = odom                # Odometry of the sonar upon swath arrival
        self.altitude = altitude        # Altitude of platform upon swath arrival


class SideScanSonar:
    def __init__(self, n_bins=1000, range=30, theta=pi/4, alpha=pi/3):

        self.n_bins = n_bins                    # Number of samples per active side and ping
        self.range = range                      # Sonar range in meters
        self.theta = theta                      # Angle from sonars y-axis to its acoustic axis 
        self.alpha = alpha                      # Angle of the transducer opening
        self.slant_resolution = range/n_bins    # Slant resolution [m] across track

class MapNode(Node):

    def __init__(self) -> None:
        super().__init__('map_node')

        self.declare_parameters(
            namespace='',
            parameters=[('processed_swath_topic', 'swath_processed'),
                        ('sonar_n_bins', 1000),
                        ('sonar_range', 30),
                        ('sonar_transducer_theta', pi/4),
                        ('sonar_transducer_alpha', pi/3),
                        ('swath_ground_range_resolution', 0.03),
                        ('swaths_per_map', 200),
                        ('map_resolution', 0.1),
                        ('processing_period', 0.001)]
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

        self.timer = self.create_timer(
            processing_period.value, self.map_generation
        )

        self.get_logger().info("Landmark detector node initialized.")

    def processed_swath_callback(self, msg):
        yaw = utility_functions.yaw_from_quaternion(
            msg.odom.pose.pose.orientation.w, 
            msg.odom.pose.pose.orientation.x, 
            msg.odom.pose.pose.orientation.y, 
            msg.odom.pose.pose.orientation.z
        )

        odom = [
            msg.odom.pose.pose.position.x,
            msg.odom.pose.pose.position.y,
            yaw
        ]

        swath = Swath(
            data_stb=np.flip(msg.data_port),
            data_port=np.flip(msg.data_stb),
            odom=odom,
            altitude=msg.altitude
        )

        self.swath_buffer.append(swath)

    def map_generation(self):

        if len(self.swath_buffer) < self.swaths_per_map.value:
            return

        # self.swath_buffer = self.swath_buffer[200:750]

        min_x = self.swath_buffer[0].odom[0]
        max_x = self.swath_buffer[0].odom[0]
        min_y = self.swath_buffer[0].odom[1]
        max_y = self.swath_buffer[0].odom[1]

        for swath in self.swath_buffer:
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
        
        echo_map, prob_map = generate_map(
            n_rows, n_colums, self.sonar.n_bins, 
            self.map_resolution.value, map_origin_x, map_origin_y, 
            self.swath_buffer, self.sonar.range, 0.5*pi/180, 
            self.swath_ground_range_resolution.value
        )

        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(echo_map, cmap='copper', vmin=0.0, vmax=1.5)

        ax2 = fig.add_subplot(1, 2, 2)
        #For probability map
        # ax2.imshow(prob_map, cmap='gray', vmin=0.0, vmax=1.0)
        # For variance map
        ax2.imshow(prob_map, cmap='copper', vmin=0.0, vmax=0.05)

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

        plt.show()
    
        input('Press any key to continue')
