class VideoSync:
    def __init__(self, left_player, right_player):
        self.left_player = left_player
        self.right_player = right_player
        self.offset_left = 0
        self.offset_right = 0

    def set_offsets(self, left_offset, right_offset):
        self.offset_left = left_offset
        self.offset_right = right_offset

    def sync_to_frame(self, frame):
        self.left_player.set_frame(frame + self.offset_left)
        self.right_player.set_frame(frame + self.offset_right)
