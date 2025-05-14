import robo_gym_server_modules.robot_server.client as rs_client
import robo_gym_server_modules.server_manager.client as sm_client


class ManagedClient(rs_client.Client):

    def __init__(
        self,
        *,
        server_manager_host: str,
        server_manager_port: int = 50100,
        launch_cmd: str,
        gui: bool = False
    ):
        if not server_manager_host:
            raise Exception("ManagedClient: missing server manager host - cannot work!")
        print(
            "Command for new simulation robot server: "
            + launch_cmd
            + " gui:="
            + ("true" if gui else "false")
        )
        self.sm_client = sm_client.Client(
            server_manager_host, server_manager_port, server_manager_port + 1
        )
        self.cmd = launch_cmd
        self.gui = gui
        self.rs_address = self.sm_client.start_new_server(cmd=self.cmd, gui=self.gui)
        super().__init__(self.rs_address)

    def kill(self):
        self.sm_client.kill_server(self.rs_address)
        # TODO might want to try async killing, too
