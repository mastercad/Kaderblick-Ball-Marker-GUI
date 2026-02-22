
class Action:
    def __init__(self, action_type: str, marker, prev_state=None):
        self.action_type = action_type  # "add", "remove", "move", "resize"
        self.marker = marker
        self.prev_state = prev_state
