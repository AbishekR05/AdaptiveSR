class FrameBuffer:
    def __init__(self):
        self._frames = {}  # Maps frame_no (int) -> enhanced_frame (np.ndarray)

    def put(self, frame_no: int, frame):
        self._frames[frame_no] = frame

    def ordered_frames(self):
        for i in sorted(self._frames.keys()):
            yield self._frames[i]
