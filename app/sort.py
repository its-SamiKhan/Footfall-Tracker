import numpy as np
from filterpy.kalman import KalmanFilter

class Sort:
    def __init__(self):
        self.trackers = []
        self.track_id = 0

    def update(self, detections):
        # Update the trackers with new detections
        if len(detections) == 0:
            return []

        # Initialize new trackers
        new_trackers = []
        for det in detections:
            x1, y1, x2, y2 = det
            new_trackers.append(self._create_tracker(x1, y1, x2, y2))

        # Update existing trackers
        for tracker in self.trackers:
            tracker.predict()

        # Add new trackers
        self.trackers.extend(new_trackers)

        # Return the current track IDs and their bounding boxes
        return [(tracker.id, tracker.get_state()) for tracker in self.trackers]

    def _create_tracker(self, x1, y1, x2, y2):
        tracker = KalmanBoxTracker(self.track_id, x1, y1, x2, y2)
        self.track_id += 1
        return tracker

class KalmanBoxTracker:
    def __init__(self, track_id, x1, y1, x2, y2):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.x[:4] = np.array([x1, y1, x2, y2])  # Initial state
        self.kf.P *= 1000.  # Initial uncertainty
        self.kf.R[2:, 2:] *= 10.  # Measurement noise
        self.kf.Q[4:, 4:] *= 1000.  # Process noise
        self.kf.F = np.array([[1, 0, 1, 0, 0, 0, 0],
                               [0, 1, 0, 1, 0, 0, 0],
                               [0, 0, 1, 0, 1, 0, 0],
                               [0, 0, 0, 1, 0, 1, 0],
                               [0, 0, 0, 0, 1, 0, 1],
                               [0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 1]])  # State transition matrix
        self.id = track_id

    def predict(self):
        self.kf.predict()

    def update(self, measurement):
        self.kf.update(measurement)

    def get_state(self):
        return self.kf.x[:4] 