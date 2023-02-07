import numpy as np
from math import pi

from rclpy.node import Node

from brov2_interfaces.msg import SonarProcessed



class Map:
    def __init__(self, height = 100, width = 100, resolution = 0.1, probability_layer = True) -> None:
        self.height = height            # Height of the map in meters
        self.width = width              # Width of the map in meters
        self.resolution = resolution    # Map resolution on meters
        self.origin= None               # The map origin in world coordinates

        # Map consisting of processed intensity returns from the sonar. 
        self.intensity_map = np.zeros(
            (height / resolution, width/resolution), 
            dtype=float
        )

        # Map where each cell corresponds to the pobability that the cell has been observed
        if probability_layer:
            self.probability_map = np.zeros(
                (height / resolution, width/resolution), 
                dtype=float
            )

class Swath:
    def __init__(self, data_port, data_stb, odom, altitude):
        self.data_port = data_port      # Port side sonar data
        self.data_stb = data_stb        # Starboard side sonar data
        self.odom = odom                # Odometry of the sonar upon swath arrival
        self.altitude = altitude        # Altitude of platform upon swath arrival

class SideScanSonar:
    def __init__(self, n_samples=1000, range=30, theta=pi/4, alpha=pi/3):

        self.n_samples = n_samples              # Number of samples per active side and ping
        self.range = range                      # Sonar range in meters
        self.theta = theta                      # Angle from sonars y-axis to its acoustic axis 
        self.alpha = alpha                      # Angle of the transducer opening
        self.slant_resolution = range/n_samples # Slant resolution [m] across track

class MapNode(Node):

    def __init__(self) -> None:
        super().__init__('map_node')

        self.declare_parameters(
            namespace='',
            parameters=[('processed_swath_topic', 'sonar_processed')]
        )

        self.swath_buffer = []      # Buffer that contains all unprocessed corrected swaths

        self.timer = self.create_timer(
            processing_period.value, self.find_landmarks
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
