import rclpy   
from brov2_map import map_node as node
   

def main(args=None):
    rclpy.init(args=args)

    map_node = node.MapNode()
    
    rclpy.spin(map_node)
    
    map_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()