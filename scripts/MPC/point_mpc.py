#! /usr/bin/env python

import rospy
import threading
from MPC_rawrawraw import MPC

from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Point, Twist
from math import atan2
import tf.transformations as tftr
from numpy import matrix, cos, arctan2, sqrt, pi, sin, cos
import numpy as np

class Task2:

    def __init__(self, MPC1):
        self.lock = threading.Lock()
        self.MPC = MPC1
        self.RATE = rospy.get_param('/rate', 50)

        self.dt = 0.0
        self.time_start = 0.0
        self.end = False

        self.pose_init = [0.0, 0.0, 0.0]
        self.flag = True

        "Desired values setup"
        # rotation matrix [4x4] from `world` frame to `body`
        self.bTw = tftr.euler_matrix(-np.pi, 0.0, 0.0, 'rxyz')

        # # in `world` frame
        # self.A = rospy.get_param('/A', 90.0)    # [degrees]
        # self.pose_des = rospy.get_param('/pose_des', [0.5, 0.0, 2.0])

        # # in 'body' frame
        # self.pose_des = self.transform_pose(self.pose_des)
        #print(self.pose_des.T)

        self.rot_z_des = 0.0


        "ROS stuff"
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1, latch=False)
        self.sub_odom = rospy.Subscriber("/odom", Odometry, self.odometry_callback)


    def transform_pose(self, pose_w):
        # in 'body' frame
        pose_des = self.bTw * np.matrix([pose_w[0], pose_w[1], pose_w[2], 0.0]).T
        return pose_des[:3]

    def odometry_callback(self, msg):
        self.lock.acquire()
        # read current robot state
        cur_position = msg.pose.pose.position
        cur_q = msg.pose.pose.orientation
        lin_vel = msg.twist.twist.linear.x
        ang_vel = msg.twist.twist.angular.z
        
        
        cur_rpy = tftr.euler_from_quaternion((cur_q.x, cur_q.y, cur_q.z, cur_q.w))  # roll pitch yaw
        cur_rot_z = cur_rpy[2]

        #print("cur_rotation", cur_rot_z)
        if self.flag:
            self.zero_pose = [cur_position.x, cur_position.y, cur_position.z]
            self.flag = False
       
        # robot state
        x_goal = 1
        y_goal = 1
        theta_goal = np.pi

        T = np.array([[cos(theta_goal), sin(theta_goal), x_goal], [-sin(theta_goal), cos(theta_goal),  y_goal], [0, 0, 1]])
        print("Matrix", T)

        #robot state
        robot_pose = np.array([cur_position.x, cur_position.y, 1])

            #robot go to zero point ``
        y = np.dot(np.linalg.inv(T), robot_pose.transpose())
        # input_x = y[0]
        # input_y = y[1]
        # input_theta = cur_rot_z
        print("NEW POS", y)
        y2mpc = [y[0], y[1], cur_rot_z]
        # errors
        # e_x = self.pose_des[0] - cur_position.x
        # e_y = self.pose_des[1] - cur_position.y
        # error_angle = self.pose_des[1] - cur_rot_z

        # distance to goal
        # dist_to_goal = sqrt(e_x**2 + e_y**2)
        # if cur_rot_z > pi:
        #     error_angle = -arctan2(e_y, e_x) + cur_rot_z - 2 * pi
        # else:
        #     error_angle = -arctan2(e_y, e_x) + cur_rot_z

        # set control
        velocity = Twist()
        self.MPC.update_coord(y2mpc, lin_vel, ang_vel)
        u = self.MPC.compute_action(self.t, y2mpc)
        print("Computed_action", u)
        # velocity.linear.x = 0.6 * dist_to_goal * cos(error_angle)
        # velocity.angular.z = 0.1 * error_angle
        velocity.linear.x = u[0] * 1.5
        velocity.angular.z = u[1] * 1.5
        self.pub_cmd_vel.publish(velocity)
        # print(e_x)
        self.lock.release()

        """
        dist_to_goal = sqrt(e_x**2 + e_y**2)
        if cur_rot_z > pi:
            error_angle = -arctan2(e_y, e_x) + cur_rot_z - 2 * pi
        else:
        error_angle = -arctan2(e_y, e_x) + cur_rot_z
        """
        #print(error_pose.T)
        """
        if not self.end and not self.flag:
            u = self.pose_controller.get_control(error_pose, self.dt)
            print(u.T)
            velocity.linear.x = u[0]
            velocity.linear.y = u[1]
            velocity.linear.z = u[2]

        #velocity.linear.y = 0.6 * dist_to_goal * cos(error_angle)
        #velocity.angular.z = -1.0 * error_angle"""
        #self.pub_cmd_vel.publish(velocity)

    def spin(self):
        rospy.loginfo('Task started!')
        rate = rospy.Rate(self.RATE)

        #time_step = 5.0
        self.end = False

        time_prev = 0.0
        self.time_start = rospy.get_time()
        while not rospy.is_shutdown():
            t = rospy.get_time() - self.time_start
            self.t = t
            self.dt = t - time_prev
            time_prev = t

            self.pose_des = self.transform_pose([1.0, 1.0, 0.0])

            
            rate.sleep()
        rospy.loginfo('Task completed!')


if __name__ == "__main__":
    MPC1 = MPC()
    rospy.init_node('task2_node')
    task1 = Task2(MPC1)
    task1.spin()