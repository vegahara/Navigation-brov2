import rclpy
from brov2_swath_processing import swath_processing_node as node

def main(args=None):
    rclpy.init(args=args)

    swath_processing_node = node.SwathProcessingNode()

    rclpy.spin(swath_processing_node)
    
    swath_processing_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
