#!/usr/bin/env python3

import numpy as np

class UR5():
    """Universal Robots UR5 utilities.

    Attributes:
        max_joint_positions (np.array): Description of parameter `max_joint_positions`.
        min_joint_positions (np.array): Description of parameter `min_joint_positions`.
        max_joint_velocities (np.array): Description of parameter `max_joint_velocities`.
        min_joint_velocities (np.array): Description of parameter `min_joint_velocities`.

    """
    def __init__(self):

        # Indexes go from shoulder pan joint to end effector
        self.max_joint_positions = np.array([6.28,6.28,6.28,6.28,6.28,6.28])
        self.min_joint_positions = - self.max_joint_positions
        self.max_joint_velocities = np.array([np.inf] * 6)
        self.min_joint_velocities = - self.max_joint_velocities

    def _ros_joint_list_to_ur5_joint_list(self,ros_thetas):
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

    def _ur_5_joint_list_to_ros_joint_list(self,thetas):
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
        """Get pose of a random point in the UR5 workspace.

        Returns:
            np.array: [x,y,z,alpha,theta,gamma] pose.

        """
        pose =  np.zeros(6)

        singularity_area = True

        # check if generated x,y,z are in singularityarea
        while singularity_area:
            # Generate random uniform sample in semisphere taking advantage of the
            # sampling rule

            # UR5 workspace radius
            # Max d = 1.892
            R =  0.900 # reduced slightly

            phi = np.random.default_rng().uniform(low= 0.0, high= 2*np.pi)
            costheta = np.random.default_rng().uniform(low= 0.0, high= 1.0) # [-1.0,1.0] for a sphere
            u = np.random.default_rng().uniform(low= 0.0, high= 1.0)

            theta = np.arccos(costheta)
            r = R * np.cbrt(u)

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            if (x**2 + y**2) > 0.085**2:
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

    def normalize_joint_values(self, joints):
        """Normalize joint position values
        
        Args:
            joints (np.array): Joint position values

        Returns:
            norm_joints (np.array): Joint position values normalized between [-1 , 1]
        """
        for i in range(len(joints)):
            if joints[i] <= 0:
                joints[i] = joints[i]/abs(self.min_joint_positions[i])
            else:
                joints[i] = joints[i]/abs(self.max_joint_positions[i])
        return joints

class UR10():
    """Universal Robots UR10 utilities.

    Attributes:
        max_joint_positions (np.array): Description of parameter `max_joint_positions`.
        min_joint_positions (np.array): Description of parameter `min_joint_positions`.
        max_joint_velocities (np.array): Description of parameter `max_joint_velocities`.
        min_joint_velocities (np.array): Description of parameter `min_joint_velocities`.

    """
    def __init__(self):

        # Indexes go from shoulder pan joint to end effector
        self.max_joint_positions = np.array([6.28,6.28,3.14,6.28,6.28,6.28])
        self.min_joint_positions = - self.max_joint_positions
        self.max_joint_velocities = np.array([np.inf] * 6)
        self.min_joint_velocities = - self.max_joint_velocities

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


    def normalize_joint_values(self, joints):
        """Normalize joint position values
        
        Args:
            joints (np.array): Joint position values

        Returns:
            norm_joints (np.array): Joint position values normalized between [-1 , 1]
        """
        for i in range(len(joints)):
            if joints[i] <= 0:
                joints[i] = joints[i]/abs(self.min_joint_positions[i])
            else:
                joints[i] = joints[i]/abs(self.max_joint_positions[i])
        return joints


    
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
