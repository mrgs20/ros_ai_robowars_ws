#! /usr/bin/python3

import rospy, message_filters
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
import angles
from tf.transformations import euler_from_quaternion
import numpy as np
from tensorflow import keras

model = keras.models.load_model('/home/mari/ros_ai_robowars_ws/saved_model/robot1_model490epochs')

predict_data = np.zeros((1,7))
print(predict_data.shape)
test_predictions = model.predict(predict_data)
print(test_predictions)

class Robot_position:
    def __init__(self):
        self.node = rospy.init_node("robot1_position_node")

        self.us1_sub = message_filters.Subscriber('/ultrasonic1', Range)
        self.us2_sub = message_filters.Subscriber('/ultrasonic2', Range)
        self.us3_sub = message_filters.Subscriber('/ultrasonic3', Range)
        self.us4_sub = message_filters.Subscriber('/ultrasonic4', Range)
        self.us5_sub = message_filters.Subscriber('/ultrasonic5', Range)
        self.us6_sub = message_filters.Subscriber('/ultrasonic6', Range)
        self.odom_sub = message_filters.Subscriber('/robot1/odom', Odometry)

        self.subs = message_filters.ApproximateTimeSynchronizer([self.us1_sub, self.us2_sub, self.us3_sub, self.us4_sub, self.us5_sub,
                                                                self.us6_sub, self.odom_sub], queue_size=1, slop=0.9, allow_headerless=True)
        self.subs.registerCallback(self.sensor_cb)
        self.predict_data = np.zeros((1,7))
        self.model = keras.models.load_model('/home/mari/ros_ai_robowars_ws/saved_model/robot1_model490epochs')

    def sensor_cb(self, us1_sub, us2_sub, us3_sub, us4_sub, us5_sub, us6_sub, odom_sub):
        orientation_in_quaternions = (
            odom_sub.pose.pose.orientation.x,
            odom_sub.pose.pose.orientation.y,
            odom_sub.pose.pose.orientation.z,
            odom_sub.pose.pose.orientation.w)

        orientation_in_euler = euler_from_quaternion(orientation_in_quaternions)
        yaw = orientation_in_euler[2]
        yaw_radians = angles.normalize_angle_positive(yaw)

        ground_truth_x = odom_sub.pose.pose.position.x
        ground_truth_y = odom_sub.pose.pose.position.y

        self.predict_data[0][0] = yaw
        self.predict_data[0][1] = us1_sub.range
        self.predict_data[0][2] = us2_sub.range
        self.predict_data[0][3] = us3_sub.range
        self.predict_data[0][4] = us4_sub.range
        self.predict_data[0][5] = us5_sub.range
        self.predict_data[0][6] = us6_sub.range

        test_predictions = model.predict(predict_data)
        print("predictions  " + str(test_predictions[0][0]) + " " + str(test_predictions[0][1]))
        print("ground truth " + str(ground_truth_x) + " " + str(ground_truth_y))

if __name__ == '__main__':
    print("start")
    Robot_position()
    rospy.spin()
    
