import pandas as pd
from math import modf

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
    for df, topic_name, msg_type, populate_fcn in zip(df_arr, topic_name_arr, msg_type_arr, populate_fcn_arr): 
        bad_msgs = 0
        good_msgs = 0
        for row in range(df.shape[0]):
            msg = msg_type()

            seconds_decimal, seconds_int = modf(df['timestamp'][row])
            time_stamp = Time(seconds = seconds_int, nanoseconds = seconds_decimal * (10 ** 9))

            msg = msg_type()
            msg, msg_good = populate_fcn(msg, df, row) 
            msg.header.stamp = time_stamp.to_msg()    
            
            if msg_good:
                good_msgs += 1
                writer.write(topic_name, serialize_message(msg), time_stamp.nanoseconds)
            else:
                bad_msgs += 1

        print("Data with topic name ", topic_name, " is written to bag")
        print("Wrote ", good_msgs, " messages to the bag")
        print("Got ", bad_msgs, " bad messages from the populate fcn")   

def populate_sonar_msg(sonar_msg, df, row):
    # Just discarding last value in data for now, should be investigated
    # Some messages has only 500 bytes and gets discarded
    
    data = df[' data'][row]
    sonar_data = []
    
    for i in range(0, len(data), 2):
        sonar_data.append(int(data[i:i+2], base=16))

    if (len(sonar_data) != 2001):
        return sonar_msg, False
    else:
        sonar_msg.data_zero = [int_val.to_bytes(2, 'big') for int_val in sonar_data[:len(sonar_data)//2]]
        sonar_msg.data_one = [int_val.to_bytes(2, 'big') for int_val in sonar_data[len(sonar_data)//2:-1]]

    return sonar_msg, True

def populate_dvl_msg(dvl_msg, df, row):
    dvl_msg.altitude  = df[' alt (m)'][row]

    return dvl_msg, True

def populate_odom_msg(odom_msg, df, row):
    phi = df[' phi (rad)'][row]
    theta = df[' theta (rad)'][row]
    psi = df[' psi (rad)'][row]

    quaternion = quaternion_from_euler(phi, theta, psi)
    w, x, y, z = quaternion

    odom_msg.pose.pose.orientation.w = w
    odom_msg.pose.pose.orientation.x = x
    odom_msg.pose.pose.orientation.y = y
    odom_msg.pose.pose.orientation.z = z

    odom_msg.pose.pose.position.x = df[' x (m)'][row]
    odom_msg.pose.pose.position.y = df[' y (m)'][row]
    odom_msg.pose.pose.position.z = df[' z (m)'][row]

    return odom_msg, True

def main():
    # Set up bag
    bag_path = 'bags/test_sonar'

    storage_options, converter_options = get_rosbag_options(bag_path)

    writer = rosbag2.SequentialWriter()
    writer.open(storage_options, converter_options)

    # Set up topics
    sonar_topic_name = 'Sonar'
    dvl_topic_name = 'DVL'
    odom_topic_name = 'Odometry'

    create_topic(writer, sonar_topic_name, 'brov2_interfaces/msg/Sonar')
    create_topic(writer, dvl_topic_name, 'brov2_interfaces/msg/DVL')
    create_topic(writer, odom_topic_name, 'nav_msgs/msg/Odometry')

    # Load data
    df_sonar = pd.read_csv('bags/SonarData.csv')
    df_dvl = pd.read_csv('bags/EstimatedState.csv') 
    df_odom = pd.read_csv('bags/EstimatedState.csv')

    convert_data(writer,
                [df_dvl, df_odom, df_sonar],  
                [dvl_topic_name, odom_topic_name, sonar_topic_name], 
                [DVL, Odometry, Sonar], 
                [populate_dvl_msg, populate_odom_msg, populate_sonar_msg]
    )

    return 0

if __name__ == "__main__":
    main()