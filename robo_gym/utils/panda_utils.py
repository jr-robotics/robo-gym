#!/usr/bin/env python3

import numpy as np
import yaml
import os  
import copy


class PANDA():
    """Panda Robot utilities class.

    Attributes:
        max_joint_positions (np.array): Maximum joint position values (rad)`.
        min_joint_positions (np.array): Minimum joint position values (rad)`.
        max_joint_velocities (np.array): Maximum joint velocity values (rad/s)`.
        min_joint_velocities (np.array): Minimum joint velocity values (rad/s)`.
        joint_names (list): Joint names (Standard Indexing)`.

    Joint Names (ROS Indexing):
    ["panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]

    """

    def __init__(self, model):

        self.model = model
        
        file_name = model + ".yaml"
        file_path = os.path.join(os.path.dirname(__file__), 'panda_parameters', file_name)

        # Load robot parameters
        with open(file_path, 'r') as stream:
            try:
                p = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc) 

        # Joint Names (Standard Indexing):
        self.joint_names = ["panda_joint1", "panda_joint2", "panda_joint3",
                            "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
        
        # Initialize joint limits attributes
        self.max_joint_positions = np.zeros(7)
        self.min_joint_positions = np.zeros(7)
        self.max_joint_velocities = np.zeros(7)
        self.min_joint_velocities = np.zeros(7)
        self.max_joint_accelerations = np.zeros(7)

        for idx,joint in enumerate(self.joint_names):
            self.max_joint_positions[idx] = p["robot_config"]["joint_config"]["joint_position_limit"]["upper"][joint]
            self.min_joint_positions[idx] = p["robot_config"]["joint_config"]["joint_position_limit"]["lower"][joint]
            self.max_joint_velocities[idx] = p["robot_config"]["joint_config"]["joint_velocity_limit"][joint]
            self.min_joint_velocities[idx] = -p["robot_config"]["joint_config"]["joint_velocity_limit"][joint]
            self.max_joint_accelerations[idx] = p["robot_config"]["joint_config"]["joint_acceleration_limit"][joint]

        # Workspace parameters
        self.ws_r = p["workspace_area"]["r"]
        self.ws_min_r = p["workspace_area"]["min_r"]

        
    def get_max_joint_positions(self):

        return self.max_joint_positions

    def get_min_joint_positions(self):

        return self.min_joint_positions

    def get_max_joint_velocities(self):

        return self.max_joint_velocities

    def get_max_joint_accelerations(self):

        return self.max_joint_accelerations

    def get_min_joint_velocities(self):

        return self.min_joint_velocities

    def normalize_joint_values(self, joints):
        """Normalize joint position values
        
        Args:
            joints (np.array): Joint position values

        Returns:
            norm_joints (np.array): Joint position values normalized between [-1 , 1]
        """
        
        joints = copy.deepcopy(joints)
        for i in range(len(joints)):
            joints[i] = (2 * joints[i] - self.max_joint_positions[i] - self.min_joint_positions[i])/(self.max_joint_positions[i] - self.min_joint_positions[i])
        return joints

    def normalize_velocity_values(self, velocites):
        """Normalize joint position values

        Args:
            joints (np.array): Joint position values

        Returns:
            norm_joints (np.array): Joint position values normalized between [-1 , 1]
        """

        vels = copy.deepcopy(velocites)
        for i in range(len(vels)):
            if vels[i] <= 0:
                vels[i] = vels[i] / abs(self.min_joint_velocities[i])
            else:
                vels[i] = vels[i] / abs(self.max_joint_velocities[i])
        return vels

    def get_random_workspace_pose(self):
        """Get pose of a random point in the robot workspace.

        Returns:
            np.array: [x,y,z,alpha,theta,gamma] pose.

        """
        pose =  np.zeros(6)

        singularity_area = True

        # check if generated x,y,z are in singularityarea
        while singularity_area:
            # Generate random uniform sample in semisphere taking advantage of the
            # sampling rule

            phi = np.random.default_rng().uniform(low= 0.0, high= 2*np.pi)
            costheta = np.random.default_rng().uniform(low= 0.0, high= 1.0) # [-1.0,1.0] for a sphere
            u = np.random.default_rng().uniform(low= 0.0, high= 1.0)

            theta = np.arccos(costheta)
            r = self.ws_r * np.cbrt(u)

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            if (x**2 + y**2) > self.ws_min_r**2:
                singularity_area = False

        pose[0:3] = [x, y, z]

        return pose

    def get_random_pose_in_front(self):
        """Get pose of a random point in the robot workspace in the area only in front of the robot.

        Returns:
            np.array: [x,y,z,alpha,theta,gamma] pose.

        """
        pose =  np.zeros(6)

        singularity_area = True

        # check if generated x,y,z are in singularityarea
        while singularity_area:
            # Generate random uniform sample in semisphere taking advantage of the
            # sampling rule

            phi = np.random.default_rng().uniform(low= -np.pi/2, high= np.pi/2)
            costheta = np.random.default_rng().uniform(low= 0.2, high= 1.0) # [-1.0,1.0] for a sphere
            u = np.random.default_rng().uniform(low= 0.0, high= 0.8)

            theta = np.arccos(costheta)
            r = self.ws_r * np.cbrt(u)

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            if (x**2 + y**2) > self.ws_min_r**2:
                singularity_area = False

        pose[0:3] = [x, y, z]

        return pose

    def _ros_joint_list_to_panda_joint_list(self, ros_thetas):
        """Transform joint angles list from ROS indexing to standard indexing.

        Rearrange a list containing the joints values from the joint indexes used
        in the ROS join_states messages to the standard joint indexing going from
        base to end effector.
        

        Args:
            ros_thetas (list): Joint angles with ROS indexing.

        Returns:
            np.array: Joint angles with standard indexing.
            
            
        NOTE: No rearrangement was needed for panda robot as the order of the joints were correct.

        """

        return np.array([ros_thetas[0],ros_thetas[1],ros_thetas[2],ros_thetas[3],ros_thetas[4],ros_thetas[5],ros_thetas[6]])

    def _panda_joint_list_to_ros_joint_list(self,thetas):
        """Transform joint angles list from standard indexing to ROS indexing.

        Rearrange a list containing the joints values from the standard joint indexing
        going from base to end effector to the indexing used in the ROS
        join_states messages.

        Args:
            thetas (list): Joint angles with standard indexing.

        Returns:
            np.array: Joint angles with ROS indexing.
            
        
        NOTE: No rearrangement was needed for panda robot as the order of the joints were correct.

        """

        return np.array([thetas[0],thetas[1],thetas[2],thetas[3],thetas[4],thetas[5],thetas[6]])
