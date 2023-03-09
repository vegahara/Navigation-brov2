import rclpy
# from rclpy.executors import MultiThreadedExecutor
# from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from brov2_landmark_detector import landmark_detector_node as node


def main(args=None):
    rclpy.init(args=args)
    
    # cb_group_subscriber = ReentrantCallbackGroup()
    # cb_group_timer = MutuallyExclusiveCallbackGroup()

    landmark_detector = node.LandmarkDetector()

    # executor = MultiThreadedExecutor()
    # executor.add_node(landmark_detector)
    
    # executor.spin()

    rclpy.spin(landmark_detector)
    
    landmark_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()