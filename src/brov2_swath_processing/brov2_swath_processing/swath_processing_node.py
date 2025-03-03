import sys
sys.path.append('utility_functions')
from utility_classes import Swath, SideScanSonar

import numpy as np
import matplotlib.pyplot as plt
from csaps import csaps
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from copy import deepcopy

from rclpy.node import Node
from rclpy.time import Time, Duration
from nav_msgs.msg import Odometry
from brov2_interfaces.msg import Sonar as SwathRaw
from brov2_interfaces.msg import SwathProcessed, DVL, SwathArray

class SwathProcessingNode(Node):

    def __init__(self):
        super().__init__('swath_data_processor')
        self.declare_parameters(namespace='', parameters=[
            ('swath_raw_topic_name', 'sonar_data'),
            ('swath_processed_topic_name', 'swath_processed'),
            ('altitude_topic_name', 'dvl/velocity_estimate'),
            ('odometry_topic_name', '/CSEI/observer/odom'),
            ('processing_period', 0.1),
            ('swath_normalizaton_smoothing_param', 1e-6),
            ('swath_ground_range_resolution', 0.03),
            ('sonar_n_bins', 1000),
            ('sonar_range', 30),
            ('sonar_transducer_theta', (25 * np.pi) / 180),
            ('sonar_transducer_alpha', np.pi/3),
            ('sonar_x_offset', -0.2532),
            ('sonar_y_offset', 0.082),
            ('sonar_z_offset', 0.033),
            ('altitude_attitude_correction', False)
        ])
            
        (swath_raw_topic_name, swath_processed_topic_name, 
        altitude_topic_name, odometry_topic_name,  
        processing_period, swath_normalizaton_smoothing_param,
        swath_ground_range_resolution,
        sonar_n_bins, sonar_range, 
        sonar_transducer_theta, sonar_transducer_alpha,
        sonar_x_offset, sonar_y_offset,
        sonar_z_offset, altitude_attitude_correction) = self.get_parameters([
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
            'sonar_transducer_alpha',
            'sonar_x_offset',
            'sonar_y_offset',
            'sonar_z_offset',
            'altitude_attitude_correction'
        ])

        # Publishers and subscribers
        self.raw_swath_sub = self.create_subscription(
            SwathRaw, swath_raw_topic_name.value, self.swath_raw_sub, 100
        )
        self.altitude_subscription   = self.create_subscription(
            DVL, altitude_topic_name.value, self.altitude_sub, 100
        )
        self.odom_subscription = self.create_subscription(
            Odometry, odometry_topic_name.value, self.odom_sub, 100
        )
        self.swath_processed_puplisher = self.create_publisher(
            SwathProcessed, swath_processed_topic_name.value, 10
            )
        self.swath_array_publisher = self.create_publisher(
            SwathArray, 'swath_array', 10
        )

        # Variable initialization
        self.swath_normalizaton_smoothing_param = swath_normalizaton_smoothing_param.value
        self.swath_ground_range_resolution = swath_ground_range_resolution.value
        self.altitude_attitude_correction = altitude_attitude_correction.value
        self.unprocessed_swaths = []
        self.unprocessed_altitudes = []
        self.unprocessed_odoms = []
        self.processed_swaths = []
        self.processed_swaths_bf = []
        self.estimation_swath_buffer = []
        self.sonar = SideScanSonar(
            sonar_n_bins.value, sonar_range.value,
            sonar_transducer_theta.value,sonar_transducer_alpha.value,
            sonar_x_offset.value, sonar_y_offset.value, sonar_z_offset.value
        )

        # Number of times we try to interpolate the pose for the newest swath before it gets discarded
        self.n_tries_interpolating_swath_limit = 10
        self.n_tries_interpolating_swath = 0 

        self.swath_array = SwathArray()

        self.timer = self.create_timer(processing_period.value, self.process_swaths)

        self.get_logger().info("Swath processing node initialized.")

    
    ### PUBLISHER AND SUBSCRIBER FUNCTIONS
    def swath_raw_sub(self, msg):

        swath = Swath(
            header = msg.header,
            data_port=np.array([int.from_bytes(b, "big") for b in msg.data_zero],dtype=float),
            data_stb=np.array([int.from_bytes(b, "big") for b in msg.data_one],dtype=float),
            odom=Odometry(),
            altitude=None
        )

        # data_stb=np.flip(np.array([int.from_bytes(b, "big") for b in msg.data_zero],dtype=float)),
        # data_port=np.flip(np.array([int.from_bytes(b, "big") for b in msg.data_one],dtype=float)),

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
        msg.data_port = swath.data_port
        msg.data_stb = swath.data_stb

        self.swath_processed_puplisher.publish(msg)

    def swath_array_pub(self, msg):

        self.swath_array_publisher.publish(msg)

    ### HELPER FUNCTIONS
    def get_corrected_sonar_altitude(self, swath:Swath, roll:float, pitch:float):
        
        if self.altitude_attitude_correction:
            altitude = swath.altitude * np.cos(roll) * np.cos(pitch)
        else:
            altitude = swath.altitude

        corr_alt_port = altitude - \
                        self.sonar.z_offset * np.cos(roll) * np.cos(pitch) + \
                        self.sonar.x_offset * np.sin(pitch) + \
                        self.sonar.y_offset * np.sin(roll) * np.cos(pitch)
        corr_alt_stb = altitude - \
                       self.sonar.z_offset * np.cos(roll) * np.cos(pitch) + \
                       self.sonar.x_offset * np.sin(pitch) - \
                       self.sonar.y_offset * np.sin(roll) * np.cos(pitch)
        
        return corr_alt_port, corr_alt_stb
    
    
    def get_range_first_bottom_return(self, swath:Swath, roll:float, pitch:float):

        corr_alt_port, corr_alt_stb = self.get_corrected_sonar_altitude(
            swath, roll, pitch
        )

        range_fbr_port = corr_alt_port / np.sin(self.sonar.theta + self.sonar.alpha/2 - roll)
        range_fbr_stb = corr_alt_stb / np.sin(self.sonar.theta + self.sonar.alpha/2 + roll)

        return range_fbr_port, range_fbr_stb
    

    def get_bin_first_bottom_return(self, swath:Swath, roll:float, pitch:float):

        range_fbr_port, range_fbr_stb = self.get_range_first_bottom_return(
            swath, roll, pitch
        )

        bin_fbr_port = int(np.floor_divide(range_fbr_port, self.sonar.slant_resolution))
        bin_fbr_stb = int(np.floor_divide(range_fbr_stb, self.sonar.slant_resolution))

        return bin_fbr_port, bin_fbr_stb


    def get_roll_pitch_yaw(self, swath:Swath):
        r = R.from_quat([
            swath.odom.pose.pose.orientation.x,
            swath.odom.pose.pose.orientation.y,
            swath.odom.pose.pose.orientation.z,
            swath.odom.pose.pose.orientation.w
        ])
        [yaw, pitch, roll] = r.as_euler('ZYX')

        return roll, pitch, yaw


    def speckle_reducing_bilateral_filter(self, data, sigma):

        # make the radius of the filter equal to 4.0 standard deviations
        radius = int(4.0 * sigma + 0.5)
        sigma2 = sigma * sigma
        x = np.arange(-radius, radius+1)
        spatial_support = np.exp(-0.5 / sigma2 * x ** 2)
        spatial_support = spatial_support / spatial_support.sum()

        filtered_data = np.zeros(data.shape)

        for i in range(len(data)):
            j = np.arange(-radius, radius+1)

            # Reflecting the borders of the signal
            j = np.where(i + j < 0, abs(j), j)
            j = np.where(i + j > len(data)-1, -j, j)

            alpha2 = data[i+j]**2
            range_support = (data[i] / alpha2) * np.exp((-data[i]**2)/(2*alpha2))
            filtered_data_point = np.sum(data[i+j] * spatial_support[j] * range_support)
            normalization_factor = np.sum(spatial_support[j] * range_support)
            filtered_data[i] = filtered_data_point / normalization_factor

        # plt.plot(data)
        # plt.plot(filtered_data)
        # plt.show()
        # input('Press any key to continue')

        return filtered_data


    def bilateral_filter(self, data, sigma_c, sigma_s, radius=None):

        # make the radius of the filter equal to 4 standard deviations
        if radius == None:
            radius = int(4.0 * sigma_c + 0.5)

        filtered_data = np.zeros(data.shape)

        for i in range(len(data)):
            j = np.arange(-radius, radius+1)

            # Reflecting the borders of the signal
            j = np.where(i + j < 0, -j, j)
            j = np.where(i + j > len(data)-1, -j, j)

            spatial_support = np.exp((-(j)**2)/(2*sigma_c**2))
            range_support = np.exp((-(data[i]-data[i+j])**2)/(2*sigma_s**2))
            filtered_data_point = np.sum(data[i+j] * spatial_support * range_support)
            normalization_factor = np.sum(spatial_support * range_support)
            filtered_data[i] = filtered_data_point / normalization_factor

        # plt.plot(data)
        # plt.plot(filtered_data)
        # plt.show()
        # input('Press any key to continue')

        return filtered_data


    ### DATA PROCESSING FUNCTIONS
    def odom_and_altitude_interpolation(self, swath:Swath):
        swath_timestamp = Time(seconds=swath.header.stamp.sec, nanoseconds=swath.header.stamp.nanosec)

        odom_new_index = None
        altitude_new_index = None

        odom_timestamp_new = None
        altitude_timestamp_new = None

        if self.unprocessed_odoms and self.unprocessed_altitudes:

            odom_timestamp_old = Time(
                seconds=self.unprocessed_odoms[0].header.stamp.sec,
                nanoseconds=self.unprocessed_odoms[0].header.stamp.nanosec
            )

            altitude_timestamp_old = Time(
                seconds=self.unprocessed_altitudes[0].header.stamp.sec,
                nanoseconds=self.unprocessed_altitudes[0].header.stamp.nanosec
            )

            if swath_timestamp - odom_timestamp_old > Duration(seconds=0, nanoseconds=0):
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

            if swath_timestamp - altitude_timestamp_old > Duration(seconds=0, nanoseconds=0):
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
        else:
            return swath, False

        # Return of we dont have odom and altitude newer than the sonar timestamp
        if (odom_new_index == None) or (altitude_new_index == None):
            self.unprocessed_altitudes = [self.unprocessed_altitudes[-1]]
            self.unprocessed_odoms = [self.unprocessed_odoms[-1]]
            return swath, False

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

        swath.odom.pose.covariance = \
            self.unprocessed_odoms[odom_new_index - 1].pose.covariance + \
            t_odom * \
            (self.unprocessed_odoms[odom_new_index].pose.covariance - \
            self.unprocessed_odoms[odom_new_index - 1].pose.covariance)
                                    
        # Remove old msgs
        self.unprocessed_odoms = self.unprocessed_odoms[odom_new_index-1:]
        self.unprocessed_altitudes = self.unprocessed_altitudes[altitude_new_index-1:]

        return swath, True

    def intensity_correction_lambertian(self, swath:Swath, roll:float, pitch:float) -> Swath:

        k_beam = 2.78

        bin_fbr_port, bin_fbr_stb = self.get_bin_first_bottom_return(
            swath, roll, pitch
        )

        corr_alt_port, corr_alt_stb = self.get_corrected_sonar_altitude(
            swath, roll, pitch
        )

        # Port swath
        swath_normalization = np.linspace(0., self.sonar.n_bins-bin_fbr_port, self.sonar.n_bins-bin_fbr_port)

        for bin in range(bin_fbr_port, self.sonar.n_bins):
            bin_slant_range = bin * self.sonar.slant_resolution
            
            insidence_angle = np.arccos(corr_alt_port / bin_slant_range)
            beam_angle = (np.pi / 2) - insidence_angle - self.sonar.theta - roll

            beam_pattern = (
                (k_beam * np.sin(beam_angle)) / \
                (np.sin( k_beam * np.sin(beam_angle))) \
            ) ** 4

            swath_normalization[-(bin - bin_fbr_port + 1)] = beam_pattern * (np.cos(insidence_angle) ** 2)

        swath.data_port[:-bin_fbr_port] = np.divide(swath.data_port[:-bin_fbr_port], swath_normalization)

        # Starboard swath
        swath_normalization = np.linspace(0., self.sonar.n_bins-bin_fbr_stb, self.sonar.n_bins-bin_fbr_stb)

        for bin in range(bin_fbr_stb, self.sonar.n_bins):
         
            bin_slant_range = bin * self.sonar.slant_resolution
            
            insidence_angle = np.arccos(corr_alt_stb / bin_slant_range)
            beam_angle = (np.pi / 2) - insidence_angle - self.sonar.theta + roll

            beam_pattern = (
                (k_beam * np.sin(beam_angle)) / \
                (np.sin( k_beam * np.sin(beam_angle))) \
            ) ** 4  

            swath_normalization[bin - bin_fbr_stb] = beam_pattern * (np.cos(insidence_angle) ** 2)

        swath.data_stb[bin_fbr_stb:] = np.divide(swath.data_stb[bin_fbr_stb:], swath_normalization)

        return swath

    def intensity_correction(self, swath:Swath, roll:float, pitch:float) -> Swath:

        bin_fbr_port, bin_fbr_stb = self.get_bin_first_bottom_return(swath, roll, pitch)

        x = np.linspace(0., self.sonar.n_bins-bin_fbr_stb, self.sonar.n_bins-bin_fbr_stb)
        spl_stb = np.flip(csaps(
            x, 
            np.flip(swath.data_stb[bin_fbr_stb:]), 
            x, 
            smooth=self.swath_normalizaton_smoothing_param
        ))

        # plt.plot(swath.data_stb)
        # plt.plot(spl_stb)
        # plt.show()
        # input('Press any key to continue')
        swath.data_stb[bin_fbr_stb:] = np.divide(swath.data_stb[bin_fbr_stb:], spl_stb)

        x = np.linspace(0., self.sonar.n_bins-bin_fbr_port, self.sonar.n_bins-bin_fbr_port)
        spl_port = csaps(
            x, 
            swath.data_port[:-bin_fbr_port], 
            x, 
            smooth=self.swath_normalizaton_smoothing_param
        )
        
        # plt.plot(swath.data_port)
        # plt.plot(spl_port)
        # plt.show()
        # input('Press any key to continue')
        
        swath.data_port[:-bin_fbr_port] = np.divide(swath.data_port[:-bin_fbr_port], spl_port)

        # plt.plot(swath.data_stb)
        # plt.show()
        # input('Press any key to continue')
        return swath
    
    def variance_normalization(self, swath:Swath, roll:float, pitch:float) -> Swath:

        window_size = 3
        h_w = window_size // 2

        bin_fbr_port, bin_fbr_stb = self.get_bin_first_bottom_return(swath, roll, pitch)

        # Port
        temp_data_port = np.empty_like(swath.data_port)
        temp_data_port[:] = np.NaN

        for bin in range(0, self.sonar.n_bins - bin_fbr_port):

            intensities = []
            r = range(max(0, bin - h_w), min(self.sonar.n_bins, bin + h_w + 1))

            for i in range(len(self.estimation_swath_buffer)-10):
                intensities.extend(self.estimation_swath_buffer[i].data_port[r])

            try:
                var = np.nanvar(intensities)
                mean = np.nanmean(intensities)
            except: 
                var = 1.0
                mean = 1.0
            temp_data_port[bin] = (swath.data_port[bin] - mean) / np.sqrt(var)

        swath.data_port = temp_data_port / 20 + 1

        # Starboard
        temp_data_stb = np.empty_like(swath.data_stb)
        temp_data_stb[:]= np.NaN

        for bin in range(bin_fbr_stb, self.sonar.n_bins):

            intensities = []
            r = range(max(0, bin - h_w), min(self.sonar.n_bins, bin + h_w + 1))

            for i in range(len(self.estimation_swath_buffer)-10):
                intensities.extend(self.estimation_swath_buffer[i].data_stb[r])

            try:
                var = np.nanvar(intensities)
                mean = np.nanmean(intensities)
            except: 
                var = 1.0
                mean = 1.0

            temp_data_stb[bin] = (swath.data_stb[bin] - mean) / np.sqrt(var)

        swath.data_stb = temp_data_stb / 20 + 1

        # # Port
        # temp_data_port = np.empty_like(swath.data_port)
        # temp_data_port[:] = np.NaN

        # for bin in range(0, self.sonar.n_bins - bin_fbr_port):

        #     r = range(max(0, bin - h_w), min(self.sonar.n_bins, bin + h_w))

        #     var = np.var(swath.data_stb[r])
        #     mean = np.mean(swath.data_stb[r])
        #     temp_data_port[bin] = (swath.data_port[bin] - mean) / np.sqrt(var)

        # swath.data_port = temp_data_port / 20 + 1

        # # Starboard
        # temp_data_stb = np.empty_like(swath.data_stb)
        # temp_data_stb[:]= np.NaN

        # for bin in range(bin_fbr_stb, self.sonar.n_bins):

        #     r = range(max(0, bin - h_w), min(self.sonar.n_bins, bin + h_w))

        #     var = np.var(swath.data_port[r])
        #     mean = np.mean(swath.data_port[r])

        #     temp_data_stb[bin] = (swath.data_stb[bin] - mean) / np.sqrt(var)

        # swath.data_stb = temp_data_stb / 20 + 1


        return swath
    

    def intensity_correction_srbf(self, swath:Swath, sigma:float) -> Swath:

        filtered_stb = self.speckle_reducing_bilateral_filter(swath.data_stb, sigma)
        swath.data_stb = np.divide(swath.data_stb, filtered_stb)

        filtered_port = self.speckle_reducing_bilateral_filter(swath.data_port, sigma)
        swath.data_port = np.divide(swath.data_port, filtered_port)

        return swath


    def intensity_correction_bf(self, swath:Swath, roll:float, pitch:float, sigma_c:float, 
                                sigma_s:float, radius:float=None) -> Swath:

        bin_fbr_port, bin_fbr_stb = self.get_bin_first_bottom_return(swath, roll, pitch)

        filtered_stb = np.flip(self.bilateral_filter(np.flip(swath.data_stb[bin_fbr_stb:]), sigma_c, sigma_s, radius))
        swath.data_stb[bin_fbr_stb:] = np.divide(swath.data_stb[bin_fbr_stb:], filtered_stb)

        filtered_port = self.bilateral_filter(swath.data_port[:-bin_fbr_port] ,sigma_c, sigma_s, radius)
        swath.data_port[:-bin_fbr_port] = np.divide(swath.data_port[:-bin_fbr_port], filtered_port)

        return swath


    def swath_filtering(self, swath:Swath, sigma:float) -> Swath:
        
        swath.data_stb = self.speckle_reducing_bilateral_filter(swath.data_stb, sigma)
        swath.data_port = self.speckle_reducing_bilateral_filter(swath.data_port, sigma)

        return swath


    def blind_zone_removal(self, swath:Swath, roll:float, pitch:float) -> Swath:
        bin_fbr_port, bin_fbr_stb = self.get_bin_first_bottom_return(
            swath, roll, pitch
        )

        swath.data_stb[:bin_fbr_stb] = [np.nan] * bin_fbr_stb
        swath.data_port[-bin_fbr_port:] = [np.nan] * bin_fbr_port

        return swath


    def slant_range_correction(self, swath:Swath, roll:float, pitch:float) -> Swath:
        # Variation of Burguera et al. 2016, Algorithm 1
    
        res = self.sonar.slant_resolution
        horisontal_y_offset = self.sonar.y_offset * np.cos(roll) + \
                              self.sonar.z_offset * np.sin(roll)
        n_bins = self.sonar.n_bins
        bin_fbr_port, bin_fbr_stb = self.get_bin_first_bottom_return(
            swath, roll, pitch
        )
        corr_alt_port, corr_alt_stb = self.get_corrected_sonar_altitude(
            swath, roll, pitch
        )

        x = np.linspace(
            0,
            self.sonar.range, 
            int(self.sonar.range / self.swath_ground_range_resolution)
        )

        ground_ranges_stb = np.array([
            horisontal_y_offset + np.sqrt((res*b)**2 - corr_alt_stb**2) for b in range(bin_fbr_stb,n_bins)
        ])
        ground_ranges_port = np.array([
            horisontal_y_offset + np.sqrt((res*b)**2 - corr_alt_port**2) for b in range(bin_fbr_port,n_bins)
        ])

        swath.data_stb = np.interp(
            x, ground_ranges_stb, swath.data_stb[bin_fbr_stb:], np.nan, np.nan
        )
        swath.data_port = np.flip(np.interp(
            x, ground_ranges_port, np.flip(swath.data_port[:-bin_fbr_port]), np.nan, np.nan
        ))

        return swath
        

    def process_swaths(self):

        if not (self.unprocessed_swaths and self.unprocessed_odoms and self.unprocessed_altitudes):
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

        self.unprocessed_swaths.pop(0)

        roll, pitch, _yaw = self.get_roll_pitch_yaw(swath)
        
        range_fbr_port, range_fbr_stb = self.get_range_first_bottom_return(swath, roll, pitch)
        if  range_fbr_port > self.sonar.range or range_fbr_stb > self.sonar.range:
            return

        # swath = self.intensity_correction_srbf(swath, 5.0)

        # swath = self.swath_filtering(swath, 1.0)

        swath = self.blind_zone_removal(swath, roll, pitch)

        # swath_bf = self.intensity_correction_bf(swath, roll, pitch, 20.0, 100.0)

        # swath = self.intensity_correction_lambertian(swath, roll, pitch)

        swath = self.intensity_correction(swath, roll, pitch)

        # if len(self.estimation_swath_buffer) >= 110:
        #     self.estimation_swath_buffer.pop(0)

        # self.estimation_swath_buffer.append(deepcopy(swath))

        # swath = self.variance_normalization(swath, roll, pitch)

        # swath = self.slant_range_correction(swath, roll, pitch)

        self.sonar_pub(swath)

        msg = SwathProcessed()
        msg.header = swath.header
        msg.odom = swath.odom
        msg.altitude = swath.altitude
        msg.data_port = swath.data_port
        msg.data_stb = swath.data_stb

        self.swath_array.swaths.append(msg)

        if len(self.swath_array.swaths) >= 50:
            self.swath_array_pub(self.swath_array)
            self.swath_array = SwathArray()

        

        # self.processed_swaths.append(np.append(swath.data_port, swath.data_stb))
        # self.processed_swaths_bf.append(np.append(swath_bf.data_port, swath_bf.data_stb))

        # if len(self.processed_swaths) > 800: 
        #     sonar_im = np.asarray(self.processed_swaths, dtype=np.float64)
            #sonar_im_bf = np.asarray(self.processed_swaths_bf, dtype=np.float64)

            # filename = '/home/repo/Navigation-brov2/images/map_waterfall_only_int_corr.csv'
            # np.savetxt(filename, sonar_im, delimiter=',')

            # input('Press any key to continue')

            # hist, _bin_edges = np.histogram(sonar_im[~np.isnan(sonar_im)], bins=200, range=(0.5,1.5))

            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

            # ax1.imshow(sonar_im, cmap='copper', vmin=0.6 , vmax=1.4)
            #ax2.imshow(sonar_im_bf, cmap='copper', vmin=0.6 , vmax=1.4)
            # ax2.plot(hist / len(sonar_im[~np.isnan(sonar_im)]))
            # ax2.set_xlim([-1, 201])
            # ax2.set_xticks(np.arange(0, 201, 20))
            # ax2.set_xticklabels([f'{x:.1f}' for x in np.arange(0.5, 1.5001, 0.1)])
            

            #plt.imshow(sonar_im, cmap='copper')
            # plt.show()

            # input('Press any key to continue')

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