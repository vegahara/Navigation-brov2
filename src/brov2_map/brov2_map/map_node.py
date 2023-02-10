import numpy as np
from math import pi

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
                        ('swaths_per_map', 1),
                        ('processing_period', 0.001)]
        )
                      
        (processed_swath_topic, 
        sonar_n_bins,
        sonar_range,
        sonar_transducer_theta,
        sonar_transducer_alpha,
        self.swaths_per_map,
        processing_period,
        ) = \
        self.get_parameters([
            'processed_swath_topic', 
            'sonar_n_bins',
            'sonar_range',
            'sonar_transducer_theta',
            'sonar_transducer_alpha',
            'swaths_per_map',
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

        swath = Swath(
            data_port=msg.data_stb,
            data_stb=msg.data_port,
            odom=msg.odom,
            altitude=msg.altitude
        )

        self.swath_buffer.append(swath)

    def map_generation(self):

        if len(self.swath_buffer) < self.swaths_per_map.value:
            return
        
        generate_map(1000, 1000, 1000, 0.01, 0, 0, self.swath_buffer, self.sonar.range, self.sonar.alpha)