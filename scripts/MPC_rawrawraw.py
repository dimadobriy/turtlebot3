from numpy.matlib import repmat
import numpy as np
from scipy.optimize import minimize
import scipy as sp

def rep_mat(argin, n, m):
    """
    Ensures 1D result

    """
    return np.squeeze(repmat(argin, n, m))

class MPC:
    def __init__(self):
        self.N = 5
        self.pred_step_size = 2
        self.dt = 0.05

        self.dim_state = 3
        self.dim_input = 2
        self.dim_output = 3
        self.dim_disturb = 2
        self._dim_full_state = self.dim_state + self.dim_disturb

        self.Vmin = -0.5
        self.Vmax = 0.5
        self.Wmin = -0.1
        self.Wmax = 0.1

        self.ctrl_bnds=np.array([[self.Vmin, self.Vmax], [self.Wmin, self.Wmax]])
        self.uMin = np.array(self.ctrl_bnds[:,0] )
        self.uMax = np.array(self.ctrl_bnds[:,1] )
        self.Umin = rep_mat(self.uMin, 1, self.N)
        self.Umax = rep_mat(self.uMax, 1, self.N)

        self.Uinit = rep_mat(self.uMin/10 , 1, self.N)
        self.uCurr = [0,0]

        self.gamma = 1
        self.k = 1

        self.t = 0

        self.input_x = 0
        self.input_y = 0
        self.input_theta = 0

        self.m = 1
        self.I = .01

    def update_coord(self, input_x, input_y, input_theta, lin_vel, ang_vel):
        self.input_x = input_x
        self.input_y = input_y
        self.input_theta = input_theta
        self.lin_vel = lin_vel
        self.ang_vel = ang_vel
        print("X_coord", self.input_x)
        print("Y_coord", self.input_y)
        print("theta", self.input_theta)
        print("lin_velocity", self.lin_vel)
        print("ang_velocity", self.ang_vel)
        
    def rcost(self, dist_to_goal, u):
        """
        Running cost (a.k.a. utility, reward, instantaneous cost etc.)

        See class documentation
        """
        chi = np.concatenate([dist_to_goal, u])
        R1 = 0.1 #np.diag([1, 0, 0, 0, 0])#self.rcost_pars[0] ->R1
        x = np.dot(chi, R1)
        r = np.dot(x, chi)

        return r

    def state_dyn(self, t, x, u):

        Dx = np.zeros(self.dim_state)
        Dx[0] = self.lin_vel * np.cos( self.input_theta )
        Dx[1] = self.lin_vel * np.sin( self.input_theta )
        Dx[2] = self.ang_vel
        

        # Dx[3] = 1/self.m * u[0]
        # Dx[4] = 1/self.I * u[1]

        return Dx

    # def closed_loop_rhs(t, ksi):
    #     """
    #     Right-hand side of the closed-loop system description.
    #     Combines everything into a single vector that corresponds to the right-hand side of the closed-loop system description for further use by simulators.
    #
    #     Attributes
    #     ----------
    #     ksi : : vector
    #         Current closed-loop system state
    #
    #     """
    #     rhs_full_state = np.zeros(self._dim_full_state)
    #
    #     x = ksi[0:self.dim_state]
    #     q = ksi[self.dim_state:]
    #
    #     # if self.ctrl_bnds.any():
    #     for k in range(self.dim_input):
    #         u[k] = np.clip(u[k], self.ctrl_bnds[k, 0], self.ctrl_bnds[k, 1])
    #
    #     rhs_full_state[:self.dim_state] = state_dyn(t, x, u)
    #
    #     return rhs_full_state

    def _actor_cost(self, U, dist_to_goal):
        """
        See class documentation. Parameter ``delta`` here is a shorthand for ``pred_step_size``

        Customization
        -------------

        Introduce your mode and the respective actor function in this method. Don't forget to provide description in the class documentation

        """

        myU = np.reshape(U, [self.N, self.dim_input]) # self.dim_input -> dim_input
        Y = np.zeros([self.N, self.dim_output]) # self.dim_input -> dim_output

        Y[0, :] = dist_to_goal
        x = [self.input_x, self.input_y, self.input_theta] #`x_sys`` represents the (true) current state of the system and should be updated accordingly.
        
        for k in range(1, self.N):
            x = x + self.pred_step_size * self.state_dyn([], x, myU[k-1, :])  # Euler scheme
            '''
            self.x_sys, sys_out(x)  
            '''
            Y[k, :] = x
        print("X_pred", x)
        J = 0
        for k in range(self.N):
            #print(Y[k,:].shape, myU[k, :].shape)
            J += self.gamma**self.k * self.rcost(Y[k, :], myU[k, :])

        return J

    def _actor(self, dist_to_goal):
        """
        See class documentation. Parameter ``delta`` here is a shorthand for ``pred_step_size``

        Customization
        -------------

        This method normally should not be altered, adjust :func:`~RLframe.controller._actor_cost`, :func:`~RLframe.controller._actor` instead.
        The only customization you might want here is regarding the optimization algorithm

        """

        actor_opt_method = 'SLSQP'
        if actor_opt_method == 'trust-constr':
            actor_opt_options = {'maxiter': 300, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            actor_opt_options = {'maxiter': 300, 'maxfev': 5000, 'disp': False, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7} # 'disp': True, 'verbose': 2}


        myUinit = np.reshape(self.Uinit, [self.N*self.dim_input,])

        bnds = sp.optimize.Bounds(self.Umin, self.Umax, keep_feasible=True)

        try:
             U = minimize(lambda U: self._actor_cost(U, dist_to_goal), myUinit, method=actor_opt_method, tol=1e-2, bounds=bnds, options=actor_opt_options).x
        except ValueError:
             print('Actor''s optimizer failed. Returning default action')
             U = myUinit
        #U = minimize(lambda U: self._actor_cost(U, dist_to_goal), myUinit, method=actor_opt_method, tol=1e-2, bounds=bnds, options=actor_opt_options).x
        return U[:self.dim_input]    # Return first action

    def compute_action(self, t, dist_to_goal):
        """
        Main method. See class documentation

        Customization
        -------------

        Add your modes, that you introduced in :func:`~RLframe.controller._actor_cost`, here

        """
        #print("distance", dist_to_goal)
        time_in_sample = t - self.t # self.ctrl_clock -> t0
        print("Time", t)
        print("Distance", dist_to_goal)
        if time_in_sample >= self.dt: # Ne
            self.t = t
            u = self._actor(dist_to_goal)
            self.uCurr = u
            print(1)
            return u

        else:
            print(2)
            return self.uCurr