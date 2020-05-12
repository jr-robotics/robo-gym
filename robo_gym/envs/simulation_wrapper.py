import robo_gym_server_modules.server_manager.client as sm_client
import robo_gym_server_modules.robot_server.client as rs_client

class Simulation:
    """Simulation Wrapper Class - can be used to add simulation capability to a robo-gym environment.

    Args:
        cmd (str): roslaunch command to execute to start the simulated Robot Server.
        ip (str): IP address of the machine hosting the Server Manager. Defaults to None.
        lower_bound_port (str): Lower bound of Server Manager port range. Defaults to None.
        upper_bound_port (str): Upper bound of Server Manager port range. Defaults to None.
        gui (bool): If True the simulation is started with GUI. Defaults to False.

    """
    def __init__(self, cmd, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):

        self.robot_server_ip = None
        self.cmd = cmd
        self.gui = gui
        if ip:
            if lower_bound_port and upper_bound_port:
                self.sm_client = sm_client.Client(ip,lower_bound_port,upper_bound_port)

            else:
                self.sm_client = sm_client.Client(ip)
            self._start_sim()

    def _start_sim(self,):
        """Start a new simulated Robot Server.
        """

        self.robot_server_ip = self.sm_client.start_new_server(cmd = self.cmd, gui = self.gui)


    def kill_sim(self,):
        """Kill the simulated Robot Server.
        """

        assert self.sm_client.kill_server(self.robot_server_ip)

    def restart_sim(self,):
        """Restart the simulated Robot Server.
        """

        self.kill_sim()
        self._start_sim()
        self.client = rs_client.Client(self.robot_server_ip)
