class InvalidStateError(Exception):
    def __init__(self, message = "The environment state received is not contained in the observation space."):
        self.message = message
    def __str__(self):
        return self.message

class RobotServerError(Exception):
    def __init__(self, service):
        if service == "set_state":
            self.message = "set_state service call failed."
        elif service == "send_action":
            self.message = "send_action service call failed."
        elif service == "get_state":
            self.message = "get_state service call failed."
        else:
            self.message = "service call failed"
    def __str__(self):
        return self.message
