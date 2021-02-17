#! /usr/bin/env python

import rospy
import threading

from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Point, Twist
from math import atan2
import tf.transformations as tftr
from numpy import matrix, cos, arctan2, sqrt, pi, sin, cos
import numpy as np


class PID:

    def __init__(self, kp, ki, kd, u_max=None, u_min=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.u_max, self.u_min = u_max, u_min
        self.counter = 0
        self.ei = np.matrix(np.zeros((3,1)))
        self.last_e = np.matrix(np.zeros((3,1)))

    def get_control(self, e, dt):

        self.counter += 1
        if self.counter == 1:
            ed = np.matrix(np.zeros((3,1)))
        else:
            self.ei += e * dt
            ed = (e - self.last_e) / dt
        self.last_e = e

        u = np.diag(self.kp, 0).dot(e) + np.diag(self.ki, 0).dot(self.ei) + np.diag(self.kd, 0).dot(ed)

        if self.u_max and self.u_min:
            if u > self.u_max:
                u = self.u_max
            elif u < self.u_min:
                u = self.u_min

        return u

class Task2:

    def __init__(self):
        self.lock = threading.Lock()
        self.RATE = rospy.get_param('/rate', 50)

        self.dt = 0.0
        self.time_start = 0.0
        self.end = False

        self.pose_init = [0.0, 0.0, 0.0]
        self.flag = True

        "Desired values setup"
        # rotation matrix [4x4] from `world` frame to `body`
        self.bTw = tftr.euler_matrix(-np.pi, 0.0, 0.0, 'rxyz')

        # in `world` frame
        self.A = rospy.get_param('/A', 90.0)    # [degrees]
        self.pose_des = rospy.get_param('/pose_des', [0.5, 0.0, 2.0])
             
        # in 'body' frame
        self.pose_des = self.transform_pose(self.pose_des)
        print(self.pose_des.T)

        self.rot_z_des = 0.0

        "Controllers"
        self.pose_controller = PID([0.1, 0.0, 0.0], 
                                   [0.0, 0.0, 0.0],
                                   [0.1, 0.0, 0.0])
        #self.orientation_controller = PID(5.0, 2.0, 0.0)

        "ROS stuff"
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1, latch=False)
        self.sub_odom = rospy.Subscriber("/odom", Odometry, self.odometry_callback)

    def rcost(y, u):
        """
        Running cost (a.k.a. utility, reward, instantaneous cost etc.)

        See class documentation
        """
        chi = np.concatenate([y, u])
        r = 0
        R1 = np.diag([1, 100, 1, 0, 0, 0, 0])#self.rcost_pars[0] ->R1
        r = np.mul(chi, R1, chi)

        return r

    def upd_icost(y, u):
        """
        Sample-to-sample integrated running cost. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``icost`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize cost instead)

        """
        dt = 0.05
        icost_val += rcost(y, u)* dt #self.sampling_time -> dt


    def _actor_cost(self, U, y, N, W, delta, mode):
        """
        See class documentation. Parameter ``delta`` here is a shorthand for ``pred_step_size``

        Customization
        -------------

        Introduce your mode and the respective actor function in this method. Don't forget to provide description in the class documentation

        """
        dim_state = 5
        dim_input = 2
        dim_output = 5
        N = 3 #horizon
        gamma = 1 #discount

        myU = np.reshape(U, [N, dim_input]) # self.dim_input -> dim_input
        Y = np.zeros([N, dim_output]) # self.dim_input -> dim_output

        Y[0, :] = y
        x = self.x_sys #`x_sys`` represents the (true) current state of the system and should be updated accordingly.
        for k in range(1, self.Nactor):
            x = x + delta * self.sys_rhs([], x, myU[k-1, :], [])  # Euler scheme
            '''
            self.x_sys, sys_out(x)  todos
            '''
            Y[k, :] = self.sys_out(x)

        J = 0
        for k in range(N):
            J += gamma**k * rcost(Y[k, :], myU[k, :])

        return J

    def _actor(y, Uinit, N, W, delta):
        """
        See class documentation. Parameter ``delta`` here is a shorthand for ``pred_step_size``

        Customization
        -------------

        This method normally should not be altered, adjust :func:`~RLframe.controller._actor_cost`, :func:`~RLframe.controller._actor` instead.
        The only customization you might want here is regarding the optimization algorithm

        """
        dim_state = 5
        dim_input = 2
        dim_output = 5
        N = 3 #horizon
        uMin = np.array(ctrl_bnds[:,0] )
        uMax = np.array(ctrl_bnds[:,1] )
        Umin = rep_mat(uMin, 1, Nactor)
        Umax = rep_mat(uMax, 1, Nactor)
        ctrl_bnds=np.array([[Fmin, Fmax], [Mmin, Mmax]])

        # Control constraints
        Fmin = -5
        Fmax = 5
        Mmin = -1
        Mmax = 1

        actor_opt_method = 'SLSQP'
        if actor_opt_method == 'trust-constr':
            actor_opt_options = {'maxiter': 300, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            actor_opt_options = {'maxiter': 300, 'maxfev': 5000, 'disp': False, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7} # 'disp': True, 'verbose': 2}

        isGlobOpt = 0

        myUinit = np.reshape(Uinit, [N*dim_input,])

        bnds = sp.optimize.Bounds(Umin, Umax, keep_feasible=True)

        try:
            if isGlobOpt:
                minimizer_kwargs = {'method': actor_opt_method, 'bounds': bnds, 'tol': 1e-7, 'options': actor_opt_options}
                U = basinhopping(lambda U: _actor_cost(U, y, N, W, delta, mode), myUinit, minimizer_kwargs=minimizer_kwargs, niter = 10).x
            else:
                U = minimize(lambda U: _actor_cost(U, y, N, W, delta, mode), myUinit, method=actor_opt_method, tol=1e-7, bounds=bnds, options=actor_opt_options).x
        except ValueError:
            print('Actor''s optimizer failed. Returning default action')
            U = myUinit
        return U[:dim_input]    # Return first action

    def compute_action(t, y):
        """
        Main method. See class documentation

        Customization
        -------------

        Add your modes, that you introduced in :func:`~RLframe.controller._actor_cost`, here

        """


        time_in_sample = t - t0 # self.ctrl_clock -> t0

        if time_in_sample >= dt: # Ne
            t0 = t
            u = _actor(y, Uinit, Nactor, [], pred_step_size, mode)
            uCurr = u
            return u

        else:
            return uCurr


    def transform_pose(self, pose_w):
        # in 'body' frame
        pose_des = self.bTw * np.matrix([pose_w[0], pose_w[1], pose_w[2], 0.0]).T
        return pose_des[:3]
            
    def odometry_callback(self, msg):
        self.lock.acquire()
        # read current robot state
        cur_position = msg.pose.pose.position
        cur_q = msg.pose.pose.orientation
        cur_rpy = tftr.euler_from_quaternion((cur_q.x, cur_q.y, cur_q.z, cur_q.w))  # roll pitch yaw
        cur_rot_z = cur_rpy[2]

        if self.flag:
            self.zero_pose = [cur_position.x, cur_position.y, cur_position.z]
            self.flag = False


        # errors
        e_x = self.pose_des[0] - cur_position.x
        e_y = self.pose_des[1] - cur_position.y
        error_angle = self.pose_des[1] - cur_rot_z

        # distance to goal
        dist_to_goal = sqrt(e_x**2 + e_y**2)
        if cur_rot_z > pi:
            error_angle = -arctan2(e_y, e_x) + cur_rot_z - 2 * pi
        else:
            error_angle = -arctan2(e_y, e_x) + cur_rot_z

        # set control

        time_in_sample = t - t0 # self.ctrl_clock -> t0

        if time_in_sample >= dt: # Ne
            t0 = t
            u = _actor(y, Uinit, Nactor, [], pred_step_size, mode)
            uCurr = u
            return u

        else:
            return uCurr

        u = _actor(y, Uinit, Nactor, [], pred_step_size, mode)
        uCurr = u
        velocity = Twist()
        velocity.linear.x = u #0.6 * dist_to_goal * cos(error_angle)
        velocity.angular.z = 0.1 * error_angle
        self.pub_cmd_vel.publish(velocity)
        #print(e_x)
        self.lock.release()


    def spin(self):
        rospy.loginfo('Task started!')
        rate = rospy.Rate(self.RATE)

        time_step = 5.0
        self.end = False

        time_prev = 0.0
        self.time_start = rospy.get_time()
        while not rospy.is_shutdown():
            t = rospy.get_time() - self.time_start
            self.dt = t - time_prev
            time_prev = t

            self.pose_des = self.transform_pose([5.0, 5.0, 0.0])
            
            #print('time: {:3.3f} dt: {:3.3f}\n'.format(t, self.dt))
            rate.sleep()
        rospy.loginfo('Task completed!')


if __name__ == "__main__":
    rospy.init_node('task2_node')
    task1 = Task2()
    task1.spin()
