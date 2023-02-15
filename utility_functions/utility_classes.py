import numpy as np


class Swath:
    def __init__(self, header, data_port, data_stb, odom, altitude):
        self.header = header            # Contains the ros msg header
        self.data_port = data_port      # Port side sonar data
        self.data_stb = data_stb        # Starboard side sonar data
        self.odom = odom                # Odometry of the sonar upon swath arrival
        self.altitude = altitude        # Altitude of platform upon swath arrival


class SideScanSonar:
    def __init__(self, n_bins=1000, range=30, theta=np.pi/4, alpha=np.pi/3):

        self.n_bins = n_bins                    # Number of samples per active side and ping
        self.range = range                      # Sonar range in meters
        self.theta = theta                      # Angle from sonars y-axis to its acoustic axis 
        self.alpha = alpha                      # Angle of the transducer opening
        self.slant_resolution = range/n_bins    # Slant resolution [m] across track


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
