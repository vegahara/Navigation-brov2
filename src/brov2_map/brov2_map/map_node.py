import sys
sys.path.append('utility_functions')
sys.path.append('utility_classes')
import utility_functions
from utility_classes import Swath, SideScanSonar

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from scipy.spatial.transform import Rotation as R

from rclpy.node import Node

from brov2_interfaces.msg import SwathProcessed

from julia.api import Julia
jl = Julia(compiled_modules=False)
jl.eval('import Pkg; Pkg.activate("src/brov2_map/brov2_map/MapGeneration")')
jl.eval('import MapGeneration')
generate_map = jl.eval('MapGeneration.MapGenerationFunctions.generate_map')

class MapNode(Node):

    def __init__(self) -> None:
        super().__init__('map_node')

        self.declare_parameters(
            namespace='',
            parameters=[('processed_swath_topic', 'swath_processed'),
                        ('sonar_n_bins', 1000),
                        ('sonar_range', 30),
                        ('sonar_transducer_theta', (25 * pi) / 180),
                        ('sonar_transducer_alpha', pi/3),
                        ('sonar_transducer_beta', (0.5*np.pi)/3),
                        ('swath_ground_range_resolution', 0.03),
                        ('swaths_per_map', 100),
                        ('map_resolution', 0.1),
                        ('processing_period', 0.001)]
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
        ])

        self.processed_swath_sub = self.create_subscription(
            SwathProcessed,
            processed_swath_topic.value,
            self.processed_swath_callback,
            qos_profile = 10
        )

        self.sonar = SideScanSonar(
            sonar_n_bins.value, sonar_range.value,
            sonar_transducer_theta.value, sonar_transducer_alpha.value,
            sonar_transducer_beta.value
        )

        self.swath_buffer = []      # Buffer that contains all unprocessed corrected swaths

        self.timer = self.create_timer(
            processing_period.value, self.map_generation
        )

        self.get_logger().info("Map node initialized.")

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

        self.swath_buffer.append(swath)

    def map_generation(self):

        if len(self.swath_buffer) < self.swaths_per_map.value:
            return

        # self.swath_buffer = self.swath_buffer[900:]

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

        print("Generating map")

        probability_threshold = 0.1

        echo_map, range_map, observed_swaths = generate_map(
            n_rows, n_colums, self.sonar.n_bins,
            self.map_resolution.value, map_origin_x, map_origin_y,
            self.swath_buffer, self.sonar.range,
            self.swath_ground_range_resolution.value, probability_threshold
        )

        # filename = '/home/repo/Navigation-brov2/images/map_400_swaths_5_cm_res_new_method.csv'
        # np.savetxt(filename, echo_map, delimiter=',')

        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(echo_map, cmap='copper', vmin=0.6, vmax=1.4)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(range_map, cmap='gray')


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
