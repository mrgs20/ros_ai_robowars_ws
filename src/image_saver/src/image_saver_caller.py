#! /usr/bin/python3

import rospy
from std_srvs.srv import Empty

rospy.init_node("image_saver_caller")
rate = rospy.Rate(0.5)

while not rospy.is_shutdown():
    rospy.wait_for_service('/camera_controller/save')
    saver = rospy.ServiceProxy('/camera_controller/save', Empty)
    saver()
    rate.sleep()

