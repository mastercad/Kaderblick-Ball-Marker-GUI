
from autosave.autosave import AUTOSAVE_SESSION_PATH
from model.action import Action

class Session:
    def __init__(self):
        self.markers = []
        self.undo_stack = []
        self.redo_stack = []
        self.videos = []
        self.timeline = []
        self.autosave_path = AUTOSAVE_SESSION_PATH

    def add_marker(self, marker):
        self.markers.append(marker)
        self.undo_stack.append(Action("add", marker))
        self.redo_stack.clear()

    def remove_marker(self, marker):
        self.markers.remove(marker)
        self.undo_stack.append(Action("remove", marker))
        self.redo_stack.clear()

    def move_marker(self, marker, new_position):
        prev_position = marker.position
        print(f"[DEBUG] move_marker: marker={marker}, prev_position={prev_position}, new_position={new_position}")
        marker.position = new_position
        self.undo_stack.append(Action("move", marker, prev_position, new_position))
        print(f"[DEBUG] move_marker: undo_stack_len={len(self.undo_stack)}")
        self.redo_stack.clear()

    def resize_marker(self, marker, new_radius):
        prev_radius = marker.radius
        marker.radius = new_radius
        self.undo_stack.append(Action("resize", marker, prev_radius, new_radius))
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return
        action = self.undo_stack.pop()
        if action.action_type == "add":
            self.markers.remove(action.marker)
        elif action.action_type == "remove":
            self.markers.append(action.marker)
        elif action.action_type == "move":
            action.marker.position = action.prev_state
        elif action.action_type == "resize":
            action.marker.radius = action.prev_state
        self.redo_stack.append(action)

    def redo(self):
        if not self.redo_stack:
            return
        action = self.redo_stack.pop()
        if action.action_type == "add":
            self.markers.append(action.marker)
        elif action.action_type == "remove":
            self.markers.remove(action.marker)
        elif action.action_type == "move":
            action.marker.position = action.new_state
        elif action.action_type == "resize":
            action.marker.radius = action.new_state
        self.undo_stack.append(action)
