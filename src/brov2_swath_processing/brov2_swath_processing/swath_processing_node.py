import sys
sys.path.append('utility_functions')
from utility_classes import Swath, SideScanSonar

import numpy as np
import matplotlib.pyplot as plt
from csaps import csaps
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from rclpy.node import Node
from rclpy.time import Time, Duration
from nav_msgs.msg import Odometry
from brov2_interfaces.msg import Sonar as SwathRaw
from brov2_interfaces.msg import SwathProcessed
from brov2_interfaces.msg import DVL

class SwathProcessingNode(Node):

    def __init__(self):
        super().__init__('sonar_data_processor')
        self.declare_parameters(namespace='', parameters=[
            ('swath_raw_topic_name', 'sonar_data'),
            ('swath_processed_topic_name', 'swath_processed'),
            ('altitude_topic_name', 'dvl/velocity_estimate'),
            ('odometry_topic_name', '/CSEI/observer/odom'),
            ('processing_period', 0.01),
            ('swath_normalizaton_smoothing_param', 1e-6),
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
        self.swath_normalizaton_smoothing_param = swath_normalizaton_smoothing_param.value
        self.swath_ground_range_resolution = swath_ground_range_resolution.value
        self.unprocessed_swaths = []
        self.unprocessed_altitudes = []
        self.unprocessed_odoms = []
        self.processed_swaths = []
        self.sonar = SideScanSonar(
            sonar_n_bins.value, sonar_range.value,
            sonar_transducer_theta.value,sonar_transducer_alpha.value
        )

        # Number of times we try to interpolate the pose for the newest swath before it gets discarded
        self.n_tries_interpolating_swath_limit = 10
        self.n_tries_interpolating_swath = 0 

        self.timer = self.create_timer(processing_period.value, self.process_swaths)

        self.get_logger().info("Swath processing node initialized.")

    
    ### PUBLISHER AND SUBSCRIBER FUNCTIONS
    def swath_raw_sub(self, msg):

        swath = Swath(
            header = msg.header,
            data_port=np.array([int.from_bytes(b, "big") for b in msg.data_zero]),
            data_stb=np.array([int.from_bytes(b, "big") for b in msg.data_one]),
            odom=Odometry(),
            altitude=None
        )

        self.unprocessed_swaths.append(swath)


    def altitude_sub(self, msg):
        # Altitude values of -1 are invalid
        if msg.altitude != -1:
            self.unprocessed_altitudes.append(msg)  


    def odom_sub(self, msg):
        self.unprocessed_odoms.append(msg)


    def sonar_pub(self, swath:Swath):
        msg = SwathProcessed()
        msg.header = swath.header
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
    def odom_and_altitude_interpolation(self, swath:Swath):
        swath_timestamp = Time(seconds=swath.header.stamp.sec, nanoseconds=swath.header.stamp.nanosec)

        odom_new_index = None
        altitude_new_index = None

        odom_timestamp_new = None
        altitude_timestamp_new = None

        if self.unprocessed_odoms and self.unprocessed_altitudes:
            for i in range(1,len(self.unprocessed_odoms)):
                odom_timestamp_new = Time(
                    seconds=self.unprocessed_odoms[i].header.stamp.sec,
                    nanoseconds=self.unprocessed_odoms[i].header.stamp.nanosec
                )
            
                if swath_timestamp - odom_timestamp_new <= Duration(seconds=0, nanoseconds=0):
                    odom_new_index = i
                    break
                else:
                    odom_timestamp_old = odom_timestamp_new

            for i in range(1,len(self.unprocessed_altitudes)):
                altitude_timestamp_new = Time(
                    seconds=self.unprocessed_altitudes[i].header.stamp.sec,
                    nanoseconds=self.unprocessed_altitudes[i].header.stamp.nanosec
                )
            
                if swath_timestamp - altitude_timestamp_new <= Duration(seconds=0, nanoseconds=0):
                    altitude_new_index = i
                    break
                else:
                    altitude_timestamp_old = altitude_timestamp_new

        # Return of we dont have odom and altitude newer than the sonar timestamp
        if odom_new_index == None or altitude_new_index == None:
            self.unprocessed_altitudes = [self.unprocessed_altitudes[-1]]
            self.unprocessed_odoms = [self.unprocessed_odoms[-1]]
            return swath, False


        odom_timestamp_old = Time(
            seconds=self.unprocessed_odoms[odom_new_index-1].header.stamp.sec,
            nanoseconds=self.unprocessed_odoms[odom_new_index-1].header.stamp.nanosec
        )

        altitude_timestamp_old = Time(
            seconds=self.unprocessed_altitudes[altitude_new_index-1].header.stamp.sec,
            nanoseconds=self.unprocessed_altitudes[altitude_new_index-1].header.stamp.nanosec
        )

        # Altitude interpolation
        t1 = swath_timestamp - altitude_timestamp_old
        t2 = altitude_timestamp_new - altitude_timestamp_old
        t_altitude = t1.nanoseconds / t2.nanoseconds

        altitude_old = self.unprocessed_altitudes[altitude_new_index-1].altitude
        altitude_new = self.unprocessed_altitudes[altitude_new_index].altitude

        swath.altitude = altitude_old + t_altitude * (altitude_new - altitude_old)
   
        # Odometry interpolation
        t1 = swath_timestamp - odom_timestamp_old
        t2 = odom_timestamp_new - odom_timestamp_old
        t_odom = t1.nanoseconds / t2.nanoseconds

        quaternions = R.from_quat([
            [
                self.unprocessed_odoms[odom_new_index-1].pose.pose.orientation.x,
                self.unprocessed_odoms[odom_new_index-1].pose.pose.orientation.y,
                self.unprocessed_odoms[odom_new_index-1].pose.pose.orientation.z,
                self.unprocessed_odoms[odom_new_index-1].pose.pose.orientation.w
            ],[
                self.unprocessed_odoms[odom_new_index].pose.pose.orientation.x,
                self.unprocessed_odoms[odom_new_index].pose.pose.orientation.y,
                self.unprocessed_odoms[odom_new_index].pose.pose.orientation.z,
                self.unprocessed_odoms[odom_new_index].pose.pose.orientation.w
            ]
        ])

        slerp = Slerp([0,1],quaternions)
        q_interpolated = slerp(t_odom)

        swath.odom.pose.pose.orientation.x = q_interpolated.as_quat()[0]
        swath.odom.pose.pose.orientation.y = q_interpolated.as_quat()[1]
        swath.odom.pose.pose.orientation.z = q_interpolated.as_quat()[2]
        swath.odom.pose.pose.orientation.w = q_interpolated.as_quat()[3]

        position_old = self.unprocessed_odoms[odom_new_index-1].pose.pose.position
        position_new = self.unprocessed_odoms[odom_new_index].pose.pose.position

        swath.odom.pose.pose.position.x = position_old.x + t_odom * (position_new.x - position_old.x)
        swath.odom.pose.pose.position.y = position_old.y + t_odom * (position_new.y - position_old.y)
        swath.odom.pose.pose.position.z = position_old.z + t_odom * (position_new.z - position_old.z)

        # Remove old msgs
        self.unprocessed_odoms = self.unprocessed_odoms[odom_new_index-1:]
        self.unprocessed_altitudes = self.unprocessed_altitudes[altitude_new_index-1:]

        return swath, True


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

        swath = self.unprocessed_swaths[0]

        self.n_tries_interpolating_swath += 1
        swath, continue_processing = self.odom_and_altitude_interpolation(swath)

        if not continue_processing:
            if self.n_tries_interpolating_swath >= self.n_tries_interpolating_swath_limit:
                self.get_logger().info("Was not able to interpolate pose for swath. Swath is discarded")
                self.unprocessed_swaths.pop(0)
            return  
        else:
            self.n_tries_interpolating_swath = 0

        range_fbr, _index_fbr = self.get_first_bottom_return(self.unprocessed_swaths[0])
        if  range_fbr > self.sonar.range:
            self.unprocessed_swaths.pop(0)
            return

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

        # self.processed_swaths.append(swath)

        # if len(self.processed_swaths) > 20:

        #     x_coordinates = []
        #     y_coordinates = []

        #     for swath in self.processed_swaths:
        #         x_coordinates.append(swath.odom.pose.pose.position.x)
        #         y_coordinates.append(swath.odom.pose.pose.position.y)

        #     plt.scatter(x_coordinates,y_coordinates, 
        #         c='grey', edgecolor='none'
        #     )
        #     plt.show()

        #     input('Press any key to continue')