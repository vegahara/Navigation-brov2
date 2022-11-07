import numpy as np
import cv2

from rclpy.node import Node

from brov2_interfaces.msg import SonarProcessed

class Swath:

    def __init__(self):
        self.swath_port = []        # Port side sonar data
        self.swath_stb = []         # Starboard side sonar data

        self.odom = None            # State of the sonar upon swath arrival
        self.altitude = None        # Altitude of platform upon swath arrival

class SideScanSonar:

    def __init__(self, nS=1000, rng=30, sensor_angle_placement=pi/4, sensor_opening=pi/3):

        self.n_samples = nS                             # Number of samples per active side and ping
        self.range = rng                                # Sonar range in meters
        self.res = (self.range*1.0)/(self.n_samples*2); # Ping resolution [m] across track. Divided by 2 to make wave traveling both back and forth.
        self.theta = sensor_angle_placement
        self.alpha = sensor_opening

class LandmarkDetector2D(Node):

    def __init__(self):
        super().__init__('landmark_detector')

        self.declare_parameters(namespace='',
            parameters=[('sonar_data_topic_name', 'sonar_processed'),
                        ('n_samples', 1000),
                        ('range_sonar', 90),
                        ('scan_lines_per_frame', 500),
                        ('processing_period', 0.001)]
        )
                      
        (sonar_data_topic_name, 
        n_samples,
        sonar_range,
        self.scan_lines_per_frame,
        processing_period) = \
        self.get_parameters([
            'sonar_data_topic_name', 
            'n_samples',
            'range_sonar',
            'scan_lines_per_frame',
            'processing_period'
        ])

        self.sonar_processed_subscription = self.create_subscription(
            SonarProcessed, 
            sonar_data_topic_name.value, 
            self.sonar_processed_callback, 
            qos_profile = 10
        )

        self.sonar = SideScanSonar(
            nS = n_samples.value,
            rng = sonar_range.value
        )

        self.swath_buffer = []       # Buffer containing swaths to process

        self.timer = self.create_timer(
            processing_period.value, self.find_landmarks
        )

        self.get_logger().info("Landmark detector node initialized.")

    def sonar_processed_callback(self, sonar_processed_msg):
        
        swath = Swath()

        swath.altitude = sonar_processed_msg.altitude
        swath.odom = sonar_processed_msg.odom

        swath.swath_stb = sonar_processed_msg.data_stb
        swath.swath_port = sonar_processed_msg.data_port

        self.swath_buffer.append(swath)

    def find_landmarks(self):
        buffer_size = len(self.swath_buffer)
        if not (buffer_size%self.scan_lines_per_frame.value == 0 and 
                buffer_size != 0):
            return

        shadow_candidates = self.find_shadow_candidates(swath_buffer)

        threshold = np.mean(swath_buffer) / 2
        print(threshold)
        ret, shadows = \
            cv2.threshold(swath_buffer, threshold, 1, cv2.THRESH_BINARY)
        print(ret)
        contours, _ = cv2.findContours(shadows, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            corr_area = area * self.d_ob_min.value / d_ob
            area_bounding_box = w * h

            if (h < self.min_height_shadow.value or 
                h > self.max_height_shadow.value or
                corr_area <= self.min_corr_area.value or
                area / area_bounding_box < self.bounding_box_fill_limit.value):
                for x, y in cnt:
                    shadows[x][y] = 0 # Remove all points thats not a shadow

          
    def find_shadow_candidates(self, swath_buffer):
        threshold = np.mean(swath_buffer) / 2

        shadow_candidates = [[0] * self.sonar.n_samples] * self.scan_lines_per_frame

        for i in range(self.scan_lines_per_frame):
            for j in range(self.sonar.n_samples):
                if swath_buffer[i][j] < threshold:
                    shadow_candidates[i][j] = 1
        
        return shadow_candidates