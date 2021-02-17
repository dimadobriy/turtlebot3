#!/usr/bin/python

import rospy
import math
import threading

from turtlesim.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist

lock = threading.Lock()

class Contoller(object):

    def __init__(self):
        rospy.init_node('controller_node')

        self.pose_sub = rospy.Subscriber('/turtle1/pose', Pose, self.pose_callback)
        self.vel_pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)

        self.goal_point = Point(5.54, 5.54, 0)
        self.robot_state = Pose()

    def pose_callback(self, cur_pose):
        lock.acquire()

        # read data from odometry
        x = cur_pose.x
        y = cur_pose.y
        theta = cur_pose.theta

        # save it as a class variable
        pose = Pose()
        pose.x = x
        pose.y = y
        pose.theta = theta
        self.robot_state = pose

        # print(pose)
        lock.release()

    def setpoint(self, goal_point):
        self.goal_point = goal_point

    def control_low(self, dt):
        #Controller's parameters
        kf = 1.0 #f - forward
        kr = 2.0 #r - rotation

        u_vx = 0
        u_wz = 0

        x = self.robot_state.x
        y = self.robot_state.y
        theta = self.robot_state.theta

        x_des = self.goal_point.x
        y_des = self.goal_point.y

        # print(x,y)
        # print(x_des,y_des)

        error_x = x_des - x
        error_y = y_des - y

        dist_to_goal = math.sqrt(error_x**2 + error_y**2)
        if theta > math.pi:
            error_angle = math.atan2(error_y, error_x) - theta + 2*math.pi
        else:
            error_angle = math.atan2(error_y, error_x) - theta

        if error_angle > math.pi:
            error_angle -= 2*math.pi
        elif error_angle < -math.pi:
            error_angle += 2*math.pi

        if abs(dist_to_goal) > 0.05:
            u_vx = kf * dist_to_goal * math.cos(error_angle)
            u_wz = kr * error_angle
 
        return u_vx, u_wz

    def spin(self):
        r = rospy.Rate(100)
        start_time = rospy.get_time()
        t_prev, dt = 0, 0
        while not rospy.is_shutdown():
            t = rospy.get_time() - start_time
            dt = t - t_prev
            t_prev = t
            # print(dt, t)

            u_vx, u_wz = self.control_low(dt)

            velocity = Twist()
            velocity.linear.x = u_vx
            velocity.angular.z = u_wz
            self.vel_pub.publish(velocity)

            r.sleep()


if __name__=="__main__":

    goal = Point()
    goal.x = 7
    goal.y = 1

    try:
        ctrl = Contoller()
        ctrl.setpoint(goal)
        ctrl.spin()
    except Exception as e:
        print(e)
