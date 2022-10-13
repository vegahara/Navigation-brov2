std_msgs/Header header

# Sonar data
byte[500] data_starboard
byte[500] data_port

# Measured altitude to the bottom (m)
float64 altitude

# pose of the sonar
geometry_msgs/Pose pose