import pandas as pd
import rclpy
import rosbag2_py
from brov2_interfaces.msg import Sonar
from brov2_interfaces.msg import DVL
from nav_msgs.msg import Odometry


df = pd.read_csv('my_csv_file.csv')

sonar_msg = Sonar()

sonar_msg.head




with rosbag.Bag('output.bag', 'w') as bag:
    for row in range(df.shape[0]):
        timestamp = rospy.Time.from_sec(df['timestamp'][row])
        imu_msg = Imu()
        imu_msg.header.stamp = timestamp

        # Populate the data elements for IMU
        # e.g. imu_msg.angular_velocity.x = df['a_v_x'][row]

        bag.write("/imu", imu_msg, timestamp)

        gps_msg = NavSatFix()
        gps_msg.header.stamp = timestamp

        # Populate the data elements for GPS

        bag.write("/gps", gpu_msg, timestamp)

        from rosidl_runtime_py.utilities import get_message

import rosbag2_py._rosbag2_py as rosbag2_py
from rclpy.serialization import deserialize_message


reader = rosbag2_py.SequentialReader()
reader.open('rosbag2_2020_01_06-14_58_37')
i = 0 
type_map = reader.get_all_topics_and_types()
while reader.has_next():
    print(f'{i}')
    i += 1
    topic, data = reader.read_next()
    msg_type = get_message(type_map[topic])
    msg = deserialize_message(data, msg_type)
    print(msg)