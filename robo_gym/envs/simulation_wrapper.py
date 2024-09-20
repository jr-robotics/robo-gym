import robo_gym_server_modules.server_manager.client as sm_client
import robo_gym_server_modules.robot_server.client as rs_client
import traceback
import random

class Simulation:
    """Simulation Wrapper Class - can be used to add simulation capability to a robo-gym environment.

    Args:
        cmd (str): roslaunch command to execute to start the simulated Robot Server.
        ip (str): IP address of the machine hosting the Server Manager. Defaults to None.
        lower_bound_port (str): Lower bound of Server Manager port range. Defaults to None.
        upper_bound_port (str): Upper bound of Server Manager port range. Defaults to None.
        gui (bool): If True the simulation is started with GUI. Defaults to False.

    """
    instances_count = 0
    verbose = False

    def __init__(self, cmd, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):

        self.robot_server_ip = None
        self.cmd = cmd
        self.gui = gui
        self.connections_count = 0
        self.expect_sim_running = False
        self.verbose = Simulation.verbose
        self.instance_id = str(chr(ord('A') + Simulation.instances_count))
        Simulation.instances_count += 1
        if ip:
            if lower_bound_port and upper_bound_port:
                self.sm_client = sm_client.Client(ip,lower_bound_port,upper_bound_port)

            else:
                self.sm_client = sm_client.Client(ip)
            self._start_sim()

    def get_instance_string(self):
        if self.robot_server_ip:
            return self.instance_id + "/" + self.robot_server_ip
        return self.instance_id

    def print(self, msg):
        if(self.verbose):
            print("Simulation " + self.get_instance_string() + ": " + str(msg))

    def _start_sim(self):
        """Start a new simulated Robot Server.
        """
        self.print("_start_sim: starting, stack trace:")
        if(self.verbose):
            traceback.print_stack()
        self.robot_server_ip = self.sm_client.start_new_server(cmd = self.cmd, gui = self.gui)
        self.connections_count += 1
        self.expect_sim_running = True
        self.print("_start_sim: end")


    def kill_sim(self):
        """Kill the simulated Robot Server.
        """
        if not self.connections_count:
            return
        if not self.expect_sim_running:
            return
        assert self.sm_client.kill_server(self.robot_server_ip)
        self.expect_sim_running = False

    def restart_sim(self):
        """Restart the simulated Robot Server.
        """

        if self.expect_sim_running:
            self.kill_sim()
        self._start_sim()
        
        # usually initialized in each environment's constructor
        self.client = rs_client.Client(self.robot_server_ip)

    def close(self):
        self.print("close start")
        try:
            if not self.expect_sim_running:
                self.print("close returning early: I don't expect my simulation to be running!")
                return
            else:
                self.print("close: I expect my simulation to be running.")                
            if not self.connections_count:
                # should be unreachable
                self.print("close returning early: I have never been connected!")
                return
            else:
                self.print("close: I have been connected " + str(self.connections_count) + " time(s)")
            if True: # hasattr(self, "sm_client") and self.sm_client is not None:
                self.print("close calling self.kill_sim")
                self.kill_sim()
                self.print("close finished self.kill_sim")
        finally:
            self.print("close end")

    def __del__(self):
        # If we reach this point, we want to kill our simulation.
        # But normal kill_sim won't work here because our client object is being removed right now, and killing takes time.
        # We can kill outside of the context of the client object if server modules are upgraded to provide this method:
        self.print("__del__ start")
        try:
            if hasattr(self, "expect_sim_running"):
                if not self.expect_sim_running:
                    self.print("__del__ returning early: I don't expect my simulation to be running!")
                    return
                else:
                    self.print("__del__: I expect my simulation to be running.")  
            else:
                self.print("__del__: I do not know if I expect my simulation to be running!")
            if hasattr(self, "connections_count"):
                if not self.connections_count:
                    self.print("__del__ returning early: I have never been connected!")
                    return
                else:
                    self.print("__del__: I have been connected " + str(self.connections_count) + " times")
            else:
                self.print("__del__: I do not know my connections count!")
            if True: # hasattr(self, "sm_client") and hasattr(sm_client.Client, 'kill_server_async') and callable(getattr(sm_client.Client, 'kill_server_async') and self.sm_client is not None):
                self.print("__del__ calling self.sm_client.kill_server_async")
                self.sm_client.kill_server_async(self.robot_server_ip)
                self.print("__del__ finished self.sm_client.kill_server_async")
        finally:
            self.print("__del__ end")