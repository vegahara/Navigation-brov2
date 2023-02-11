import numpy as np
import matplotlib.pyplot as plt
from csaps import csaps

from rclpy.node import Node
from nav_msgs.msg import Odometry
from brov2_interfaces.msg import Sonar as SwathRaw
from brov2_interfaces.msg import SwathProcessed
from brov2_interfaces.msg import DVL


class Swath:
    def __init__(self, data_port, data_stb, odom, altitude):
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


class SwathProcessingNode(Node):

    def __init__(self):
        super().__init__('sonar_data_processor')
        self.declare_parameters(namespace='', parameters=[
            ('swath_raw_topic_name', 'sonar_data'),
            ('swath_processed_topic_name', 'swath_processed'),
            ('altitude_topic_name', 'dvl/velocity_estimate'),
            ('odometry_topic_name', '/CSEI/observer/odom'),
            ('processing_period', 0.0001),
            ('swath_normalizaton_smoothing_param', 1e-8),
            ('swath_ground_range_resolution', 0.03),
            ('sonar_n_bins', 1000),
            ('sonar_range', 30),
            ('sonar_transducer_theta', np.pi/4),
            ('sonar_transducer_alpha', np.pi/3),
        ])
            
        (swath_raw_topic_name, swath_processed_topic_name, 
        altitude_topic_name, odometry_topic_name,  
        processing_period, swath_normalizaton_smoothing_param,
        swath_ground_range_resolution,
        sonar_n_bins, sonar_range, 
        sonar_transducer_theta,
        sonar_transducer_alpha) = self.get_parameters([
            'swath_raw_topic_name',
            'swath_processed_topic_name', 
            'altitude_topic_name',
            'odometry_topic_name',
            'processing_period',
            'swath_normalizaton_smoothing_param',
            'swath_ground_range_resolution',
            'sonar_n_bins',
            'sonar_range',
            'sonar_transducer_theta',
            'sonar_transducer_alpha'
        ])

        # Publishers and subscribers
        self.raw_swath_sub = self.create_subscription(
            SwathRaw, swath_raw_topic_name.value, self.swath_raw_sub, 10
        )
        self.altitude_subscription   = self.create_subscription(
            DVL, altitude_topic_name.value, self.altitude_sub, 10
        )
        self.odom_subscription = self.create_subscription(
            Odometry, odometry_topic_name.value, self.odom_sub, 10
        )
        self.swath_processed_puplisher  = self.create_publisher(
            SwathProcessed, swath_processed_topic_name.value, 10
            )

        # Variable initialization
        self.odom_initialized = False
        self.altitude_valid = False
        self.swath_normalizaton_smoothing_param = swath_normalizaton_smoothing_param.value
        self.swath_ground_range_resolution = swath_ground_range_resolution.value
        self.current_altitude = None
        self.current_odom = Odometry()
        self.unprocessed_swaths = []
        self.processed_swaths = []
        self.sonar = SideScanSonar(
            sonar_n_bins.value, sonar_range.value,
            sonar_transducer_theta.value,sonar_transducer_alpha.value
        )

        self.timer = self.create_timer(processing_period.value, self.process_swaths)

        self.get_logger().info("Swath processing node initialized.")

    
    ### PUBLISHER AND SUBSCRIBER FUNCTIONS
    def swath_raw_sub(self, msg):

        # Ignore swath if we dont have odometry and valid altitude
        if not self.odom_initialized or not self.altitude_valid:
            return

        swath = Swath(
            data_port=np.array([int.from_bytes(b, "big") for b in msg.data_zero]),
            data_stb=np.array([int.from_bytes(b, "big") for b in msg.data_one]),
            odom=self.current_odom,
            altitude=self.current_altitude
        )

        self.unprocessed_swaths.append(swath)


    def altitude_sub(self, msg):
        # Altitude values of -1 are invalid
        if msg.altitude != -1:
            self.altitude_valid = True
        else:
            self.altitude_valid = False
        self.current_altitude = msg.altitude


    def odom_sub(self, msg):
        self.current_odom = msg
        self.odom_initialized = True


    def sonar_pub(self, swath:Swath):
        msg = SwathProcessed()
        msg.header = swath.odom.header
        msg.odom = swath.odom
        msg.altitude = swath.altitude
        msg.data_stb = swath.data_stb
        msg.data_port = swath.data_port

        self.swath_processed_puplisher.publish(msg)


    ### HELPER FUNCTIONS
    def get_first_bottom_return(self, swath:Swath):
        range_fbr =  swath.altitude / np.sin(self.sonar.theta + self.sonar.alpha/2)
        bin_number_fbr = int(np.floor_divide(range_fbr, self.sonar.slant_resolution))

        return range_fbr, bin_number_fbr


    ### DATA PROCESSING FUNCTIONS
    def intensity_correction(self, swath:Swath) -> Swath:
        x = np.linspace(0., self.sonar.n_bins, self.sonar.n_bins)
        spl_stb = csaps(x, swath.data_stb, x, smooth=self.swath_normalizaton_smoothing_param)
        swath.data_stb = np.divide(swath.data_stb, spl_stb)

        x = np.linspace(0., self.sonar.n_bins, self.sonar.n_bins)
        spl_port = csaps(x, swath.data_port, x, smooth=self.swath_normalizaton_smoothing_param)
        swath.data_port = np.divide(swath.data_port, spl_port)

        return swath


    def blind_zone_removal(self, swath:Swath) -> Swath:
        range_fbr, index_fbr = self.get_first_bottom_return(swath)

        swath.data_stb[:index_fbr] = [np.nan] * index_fbr
        swath.data_port[-index_fbr:] = [np.nan] * index_fbr

        return swath


    def slant_range_correction(self, swath:Swath) -> Swath:
        # Variation of Burguera et al. 2016, Algorithm 1

        res = self.sonar.slant_resolution
        alt = swath.altitude
        n_bins = self.sonar.n_bins
        _range_fbr, index_fbr = self.get_first_bottom_return(swath)

        x = np.linspace(
            0,
            self.sonar.range, 
            int(self.sonar.range / self.swath_ground_range_resolution)
        )
        ground_ranges = np.array([np.sqrt((res*b)**2 - alt**2) for b in range(index_fbr,n_bins)])
        swath.data_stb = np.interp(
            x, ground_ranges, swath.data_stb[index_fbr:], np.nan, np.nan
        )
        swath.data_port = np.flip(np.interp(
            x, ground_ranges, np.flip(swath.data_port[:-index_fbr]), np.nan, np.nan
        ))

        return swath
        

    def process_swaths(self):

        if len(self.unprocessed_swaths) == 0:
            return

        range_fbr, _index_fbr = self.get_first_bottom_return(self.unprocessed_swaths[0])
        if  range_fbr > self.sonar.range:
            self.unprocessed_swaths.pop(0)
            return

        swath = self.unprocessed_swaths[0]  

        swath = self.intensity_correction(swath)

        swath = self.blind_zone_removal(swath)

        swath = self.slant_range_correction(swath)

        self.unprocessed_swaths.pop(0)

        self.sonar_pub(swath)

        # self.processed_swaths.append(np.append(swath.data_port, swath.data_stb))

        # if len(self.processed_swaths) > 2990: 
        #     sonar_im = np.asarray(self.processed_swaths, dtype=np.float64)

        #     plt.imshow(sonar_im, cmap='copper', vmin=0.6,vmax=1.5)
        #     plt.show()

        #     input('Press any key to continue')