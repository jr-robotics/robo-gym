#!/usr/bin/env python3

import numpy as np

def dh_trans_matrix(theta, a, d, alpha):
    """Get Denavit-Hartemberg transformation matrix from dh parameters.

    Args:
        theta (float): theta DH parameter.
        a (float): a DH parameter.
        d (float): d DH parameter.
        alpha (float): alpha DH parameter.

    Returns:
        np.array: DH transformation matrix.

    """

    t = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)], \
                  [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)], \
                  [0            ,  np.sin(alpha)              ,  np.cos(alpha)              , d              ], \
                  [0            ,  0                          ,  0                          , 1              ]])
    return t

class UR10():
    """Universal Robots UR10 utilities.

    Attributes:
        dh (list[dict]): Robot's Denavit-Hartemberg parameters.
        max_joint_positions (np.array): Description of parameter `max_joint_positions`.
        min_joint_positions (np.array): Description of parameter `min_joint_positions`.
        max_joint_velocities (np.array): Description of parameter `max_joint_velocities`.
        min_joint_velocities (np.array): Description of parameter `min_joint_velocities`.

    """
    def __init__(self):

        self.dh = [{'theta':0, 'a':0,       'd':0.1273,     'alpha': np.pi/2},
                   {'theta':0, 'a':-0.612,  'd':0,          'alpha': 0},
                   {'theta':0, 'a':-0.5723, 'd':0,          'alpha': 0},
                   {'theta':0, 'a':0,       'd':0.163941,   'alpha': np.pi/2},
                   {'theta':0, 'a':0,       'd':0.1157,     'alpha': -np.pi/2},
                   {'theta':0, 'a':0,       'd':0.0922,     'alpha': 0}]

        self.max_joint_positions = np.array([3.14,6.28,6.28,6.28,6.28,6.28])
        self.min_joint_positions = - self.max_joint_positions
        self.max_joint_velocities = np.array([np.inf] * 6)
        self.min_joint_velocities = - self.max_joint_velocities


    def get_ee_pose(self,thetas):
        """Get end effector pose given the joints angles.

        Args:
            thetas (list): Joints angles (rad).

        Returns:
            list: Effector coordinates.

        """


        t = self._get_full_t(thetas)
        x = t[0][3]
        y = t[1][3]
        z = t[2][3]

        return [x,y,z]

    def _get_dh_trans_matrix(self, theta, joint_number):
        """Get Denavit-Hartemberg transformation matrix.

        Args:
            theta (float): Joint angle (rad).
            joint_number (int): Joint number.

        Returns:
            np.array: DH transformation matrix.

        """

        return dh_trans_matrix(theta, self.dh[joint_number]['a'], \
                               self.dh[joint_number]['d'], self.dh[joint_number]['alpha'])

    def _get_full_t(self, thetas):
        """Get full robot's transformation matrix.

        Args:
            thetas (list): Joints angles (rad).

        Returns:
            np.array: Full robot's transformation matrix.

        """

        t = self._get_dh_trans_matrix(thetas[0],0)
        for i in range(1,len(thetas)):
            t = np.matmul(t, self._get_dh_trans_matrix(thetas[i],i))

        return t

    def _ros_joint_list_to_ur10_joint_list(self,ros_thetas):
        """Transform joint angles list from ROS indexing to standard indexing.

        Rearrange a list containing the joints values from the joint indexes used
        in the ROS join_states messages to the standard joint indexing going from
        base to end effector.

        Args:
            ros_thetas (list): Joint angles with ROS indexing.

        Returns:
            np.array: Joint angles with standard indexing.

        """

        return np.array([ros_thetas[2],ros_thetas[1],ros_thetas[0],ros_thetas[3],ros_thetas[4],ros_thetas[5]])

    def _ur_10_joint_list_to_ros_joint_list(self,thetas):
        """Transform joint angles list from standard indexing to ROS indexing.

        Rearrange a list containing the joints values from the standard joint indexing
        going from base to end effector to the indexing used in the ROS
        join_states messages.

        Args:
            thetas (list): Joint angles with standard indexing.

        Returns:
            np.array: Joint angles with ROS indexing.

        """

        return np.array([thetas[2],thetas[1],thetas[0],thetas[3],thetas[4],thetas[5]])

    def get_random_workspace_pose(self):
        """Get pose of a random point in the UR10 workspace.

        Returns:
            np.array: [x,y,z,alpha,theta,gamma] pose.

        """
        pose =  np.zeros(6)

        singularity_area = True

        # check if generated x,y,z are in singularityarea
        while singularity_area:
            # Generate random uniform sample in semisphere taking advantage of the
            # sampling rule

            # UR10 workspace radius
            # Max d = 2.547
            R =  1.200 # reduced slightly

            phi = np.random.default_rng().uniform(low= 0.0, high= 2*np.pi)
            costheta = np.random.default_rng().uniform(low= 0.0, high= 1.0) # [-1.0,1.0] for a sphere
            u = np.random.default_rng().uniform(low= 0.0, high= 1.0)

            theta = np.arccos(costheta)
            r = R * np.cbrt(u)

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            if (x**2 + y**2) > 0.095**2:
                singularity_area = False

        pose[0:3] = [x,y,z]

        return pose

    def get_max_joint_positions(self):

        return self.max_joint_positions

    def get_min_joint_positions(self):

        return self.min_joint_positions

    def get_max_joint_velocities(self):

        return self.max_joint_velocities

    def get_min_joint_velocities(self):

        return self.min_joint_velocities
