import rclpy
from brov2_landmark_detector import landmark_detector_node as node

def main(args=None):
    rclpy.init(args=args)
    
    landmark_detector = node.LandmarkDetector()

    rclpy.spin(landmark_detector)
    
    landmark_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()