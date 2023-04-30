import sys
sys.path.append('utility_functions')
sys.path.append('utility_classes')
import utility_functions
from utility_classes import Swath, SideScanSonar

import numpy as np
from math import pi
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

from rclpy.node import Node
from rclpy.parameter import Parameter

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
                        ('swaths_per_map', 400),
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

        map_origin_x = np.ceil(max_x) + self.sonar.range
        map_origin_y = np.floor(min_y) - self.sonar.range
        n_rows = int(np.ceil(
            (max_x - min_x + 2.0 * self.sonar.range + 1.0) / self.map_resolution.value
        ))
        n_colums = int(np.ceil(
            (max_y - min_y + 2.0 * self.sonar.range + 1.0) / self.map_resolution.value
        ))

        print("Generating map")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Bitstream Vera Sans'
        plt.rcParams['font.size'] = 12
        plt.rcParams['image.aspect'] = 'equal'

        save_folder = '/home/repo/Navigation-brov2/images/map_generation/'
        methods = ['knn', 'knn', 'optimized', 'optimized', 'optimized', 'original']
        probability_thresholds = [0.1, 0.1, 0.1, 0.05, 0.1, 0.1]
        knn_ks = [4, 4, 2, 2, 2, 2]
        map_resolutions = [0.05, 0.1, 0.1, 0.1, 0.2, 0.1]

        for method, probability_threshold, knn_k, map_resolution in zip (methods, probability_thresholds, knn_ks, map_resolutions):

            self.map_resolution = Parameter('map_resolution', Parameter.Type.DOUBLE, map_resolution)

            n_rows = int(np.ceil(
                (max_x - min_x + 2.0 * self.sonar.range + 1.0) / self.map_resolution.value
            ))
            n_colums = int(np.ceil(
                (max_y - min_y + 2.0 * self.sonar.range + 1.0) / self.map_resolution.value
            ))

            intensity_map, probability_map, intensity_variance, range_map= generate_map(
                n_rows, n_colums, self.sonar.n_bins,
                self.map_resolution.value, map_origin_x, map_origin_y,
                deepcopy(self.swath_buffer), self.sonar.range,
                self.swath_ground_range_resolution.value,
                probability_threshold, knn_k, method
            )

            map_origin = [map_origin_x, map_origin_y]

            self.plot_maps(intensity_map, probability_map, intensity_variance, map_origin,
                        probability_threshold, knn_k, method, save_folder=save_folder
            )

        plt.show()

        # filename = '/home/repo/Navigation-brov2/images/map_400_swaths_5_cm_res_new_method.csv'
        # np.savetxt(filename, echo_map, delimiter=',')


    def plot_maps(self, intensity_map, probability_map, intensity_variance, map_origin,
                  probability_threshold, knn_k, method, save_folder=None):
                
        if method == 'knn':
            s = method + ' res=' + str(self.map_resolution.value) + ' knn_k=' + str(knn_k)
        else:
            s = method + ' p=' + str(probability_threshold) + ' res=' + str(self.map_resolution.value) + ' knn_k=' + str(knn_k)
        
        cmap_copper = matplotlib.cm.copper
        cmap_copper.set_bad('w', 1.)
        cmap_gray = matplotlib.cm.gray
        cmap_gray.set_bad('w', 1.)

        self.plot_map(
            intensity_map, 'Intensity map - ' + s, cmap_copper, 
            0.6, 1.4, map_origin, save_folder=save_folder
        )

        self.plot_map(
            intensity_variance, 'Intensity variance - ' + s, cmap_copper, 
            0.0, 0.05, map_origin, save_folder=save_folder
        )

        if method != 'knn':
            self.plot_map(
                probability_map, 'Probability map - ' + s, cmap_gray, 
                0.0, 1.0, map_origin, save_folder=save_folder
            )

    
    def plot_map(self, map, title, cmap, vmin, vmax, map_origin, tick_distanse=20,  save_folder=None):
        
        fig = plt.figure(title)

        plt.imshow(map, cmap=cmap, vmin=vmin, vmax=vmax)

        n_rows, n_colums = map.shape

        x_labels = []
        x_locations = []
        y_labels = []
        y_locations = []

        x_tick_start = int((map_origin[0] % tick_distanse) / self.map_resolution.value)
        y_tick_start = int((tick_distanse - map_origin[1] % tick_distanse) / self.map_resolution.value)

        for i in range(x_tick_start, n_rows, int(tick_distanse/self.map_resolution.value)):
            v = map_origin[0] - i * self.map_resolution.value
            x_labels.append(('%.2f' % v) + ' m')
            x_locations.append(i)
        for i in range(y_tick_start, n_colums, int(tick_distanse/self.map_resolution.value)):
            v = map_origin[1] + i * self.map_resolution.value
            y_labels.append(('%.2f' % v) + ' m')
            y_locations.append(i)

        plt.yticks(x_locations, x_labels)
        plt.xticks(y_locations, y_labels)
        plt.ylabel('North')
        plt.xlabel('East')

        plt.grid(visible=True)

        if save_folder != None:
            plt.savefig(save_folder + title.replace(' ', '_') + '.eps', format='eps')
        
