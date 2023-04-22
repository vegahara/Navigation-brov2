import pandas as pd
from math import modf, nan

import rosbag2_py as rosbag2
from rclpy.serialization import serialize_message
from rclpy.time import Time

from brov2_interfaces.msg import Sonar
from brov2_interfaces.msg import DVL
from nav_msgs.msg import Odometry

from utility_functions import quaternion_from_euler

import matplotlib.pyplot as plt


def get_rosbag_options(path, serialization_format='cdr'):
    storage_options = rosbag2.StorageOptions(uri=path, storage_id='sqlite3')

    converter_options = rosbag2.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options


def create_topic(writer, topic_name, topic_type, serialization_format='cdr'):

    topic_name = topic_name
    topic = rosbag2.TopicMetadata(name=topic_name, type=topic_type,
                                     serialization_format=serialization_format)

    writer.create_topic(topic)
        
def convert_data(writer, df_arr, topic_name_arr, msg_type_arr, populate_fcn_arr):
    for dfs, topic_name, msg_type, populate_fcn in zip(df_arr, topic_name_arr, msg_type_arr, populate_fcn_arr): 
        bad_msgs = 0
        good_msgs = 0

        for row in range(dfs[0].shape[0]):
            msg = msg_type()

            seconds_decimal, seconds_int = modf(dfs[0]['timestamp'][row])
            time_stamp = Time(seconds = seconds_int, nanoseconds = seconds_decimal * (10 ** 9))

            msg = msg_type()
            msg, msg_good = populate_fcn(msg, dfs, row) 
            msg.header.stamp = time_stamp.to_msg()    
            
            if msg_good:
                good_msgs += 1
                writer.write(topic_name, serialize_message(msg), time_stamp.nanoseconds)
            else:
                bad_msgs += 1

        print("Data with topic name ", topic_name, " is written to bag")
        print("Wrote ", good_msgs, " messages to the bag")
        print("Got ", bad_msgs, " bad messages from the populate fcn")   

def populate_sonar_msg(sonar_msg, dfs, row):
    df = dfs[0]
    
    if not (df[' entity '][row] == ' Sidescan'):
        return sonar_msg, False

    data = df[' data'][row].strip()
    sonar_data = []
    
    for i in range(0, len(data), 2):
        # Data is big endian and has to be interpred in reverse
        sonar_data.append(int(data[i:i+2], base=16)) 

    sonar_msg.data_zero = [int_val.to_bytes(1, 'big') for int_val in sonar_data[:len(sonar_data)//2]]
    sonar_msg.data_one = [int_val.to_bytes(1, 'big') for int_val in sonar_data[len(sonar_data)//2:]]

    return sonar_msg, True

def populate_dvl_msg(dvl_msg, dfs, row):
    df = dfs[0]
    dvl_msg.altitude  = df[' alt (m)'][row]

    return dvl_msg, True

def populate_odom_msg(odom_msg, dfs, row):
    df_odom = dfs[0]
    df_uncertainty = dfs[1]

    phi = df_odom[' phi (rad)'][row]
    theta = df_odom[' theta (rad)'][row]
    psi = df_odom[' psi (rad)'][row]

    quaternion = quaternion_from_euler(phi, theta, psi)
    w, x, y, z = quaternion

    odom_msg.pose.pose.orientation.w = w
    odom_msg.pose.pose.orientation.x = x
    odom_msg.pose.pose.orientation.y = y
    odom_msg.pose.pose.orientation.z = z

    odom_msg.pose.pose.position.x = df_odom[' x (m)'][row]
    odom_msg.pose.pose.position.y = df_odom[' y (m)'][row]
    odom_msg.pose.pose.position.z = df_odom[' z (m)'][row]

    odom_msg.twist.twist.linear.x = df_odom[' u (m/s)'][row]
    odom_msg.twist.twist.linear.y = df_odom[' v (m/s)'][row]
    odom_msg.twist.twist.linear.z = df_odom[' w (m/s)'][row]

    odom_msg.twist.twist.angular.x = df_odom[' p (rad/s)'][row]
    odom_msg.twist.twist.angular.y = df_odom[' q (rad/s)'][row]
    odom_msg.twist.twist.angular.z = df_odom[' r (rad/s)'][row]

    covariance_pose = [nan for j in range(36)]
    covariance_twist = [nan for j in range(36)]

    covariance_pose[0] = float(df_uncertainty[' x (m)'][row])
    covariance_pose[7] = float(df_uncertainty[' y (m)'][row])
    covariance_pose[14] = float(df_uncertainty[' z (m)'][row])

    covariance_pose[21] = float(df_uncertainty[' phi (rad)'][row])
    covariance_pose[28] = float(df_uncertainty[' theta (rad)'][row])
    covariance_pose[35] = float(df_uncertainty[' psi (rad)'][row])

    covariance_twist[0] = float(df_uncertainty[' u (m/s)'][row])
    covariance_twist[7] = float(df_uncertainty[' v (m/s)'][row])
    covariance_twist[14] = float(df_uncertainty[' w (m/s)'][row])

    covariance_twist[21] = float(df_uncertainty[' p (rad/s)'][row])
    covariance_twist[28] = float(df_uncertainty[' q (rad/s)'][row])
    covariance_twist[35] = float(df_uncertainty[' r (rad/s)'][row])

    odom_msg.pose.covariance = covariance_pose
    odom_msg.twist.covariance = covariance_twist

    return odom_msg, True

def main():

    # Set up bag
    bag_path = 'bag_test/'
    csv_folder = 'csv_test/'

    storage_options, converter_options = get_rosbag_options(bag_path)

    writer = rosbag2.SequentialWriter()
    writer.open(storage_options, converter_options)

    # Set up topics
    sonar_topic_name = 'sonar_data'
    dvl_topic_name = 'dvl/velocity_estimate'
    odom_topic_name = '/CSEI/observer/odom'

    create_topic(writer, sonar_topic_name, 'brov2_interfaces/msg/Sonar')
    create_topic(writer, dvl_topic_name, 'brov2_interfaces/msg/DVL')
    create_topic(writer, odom_topic_name, 'nav_msgs/msg/Odometry')

    # Load data
    df_sonar = pd.read_csv(csv_folder + 'SonarData.csv')
    df_dvl = pd.read_csv(csv_folder + 'EstimatedState.csv') 
    df_odom = pd.read_csv(csv_folder + 'EstimatedState.csv')
    df_uncertainty = pd.read_csv(csv_folder + 'NavigationUncertainty.csv')

    convert_data(writer,
                [[df_dvl], [df_odom, df_uncertainty], [df_sonar]],  
                [dvl_topic_name, odom_topic_name, sonar_topic_name], 
                [DVL, Odometry, Sonar], 
                [populate_dvl_msg, populate_odom_msg, populate_sonar_msg]
    )

    return 0

if __name__ == "__main__":
    main()