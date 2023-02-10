std_msgs/Header header

# Sonar data
float64[3000] data_stb
float64[3000] data_port

# Measured altitude to the bottom (m)
float64 altitude

# Odometry of the sonar
nav_msgs/Odometry odom