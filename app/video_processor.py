import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import time
import threading
import os
from .density_analyzer import DensityAnalyzer
from .heatmap_generator import HeatmapGenerator
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class VideoProcessor:
    def __init__(self):
        # Use the smallest YOLO model for better performance
        self.model = YOLO('yolov8n.pt')
        self.storage_path = "static/heatmaps/"
        os.makedirs(self.storage_path, exist_ok=True)  # Ensure the directory exists
        self.trackers = []
        self.track_ids = []
        
        # Performance optimization parameters
        self.frame_skip = 2  # Process every nth frame
        self.target_width = 640  # Reduced resolution width
        self.target_height = 480  # Reduced resolution height
        self.heatmap_update_interval = 2.0  # Update heatmap every n seconds
        self.last_heatmap_update = time.time()

        # Additional tracking parameters
        self.min_detection_confidence = 0.5
        self.min_tracking_confidence = 0.3
        self.max_disappeared = 30  # Frames before removing track
        self.min_distance = 50     # Minimum pixels between centroids
        self.min_frames = 3        # Minimum frames to confirm as valid person

        self.should_stop = False
        self.processing_thread = None
        self.current_count = 0
        self.total_cumulative_count = 0
        self.zone_counts = {}
        self.lock = threading.Lock()  # Thread-safe counter updates
        self.processing_complete = False

        self.zones = {}
        self.zone_data = {}  # Store detection data per zone
        self.heatmap_points = {}  # Store points for heatmap per zone
        
        # Ensure static directory exists
        os.makedirs('static/heatmaps', exist_ok=True)

        self.footfall_per_second = []
        self.active_footfall_count = 0  # Initialize active footfall count

    def process_stream(self, video_url, zones):
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(
            target=self._process_stream_internal,
            args=(video_url, zones)
        )
        self.processing_thread.start()
        return self.processing_thread

    def _process_stream_internal(self, video_url, zones):
        if video_url == "0":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_url)

        if not cap.isOpened():
            print(f"Error: Unable to open video stream {video_url}")
            return self._get_current_stats()

        # Initialize counters and data structures
        self.current_count = 0
        self.total_cumulative_count = 0
        self.zone_counts = {zone['zone_id']: 0 for zone in zones}
        heatmap_data = []
        last_time = time.time()
        frame_count = 0
        
        # Tracking state
        tracked_objects = {}       # {id: {'centroid': (x,y), 'frames': n, 'disappeared': n}}
        next_object_id = 1
        confirmed_ids = set()      # Set of IDs that have been confirmed as valid people

        try:
            while cap.isOpened() and not self.should_stop:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % self.frame_skip != 0:
                    continue

                frame = cv2.resize(frame, (self.target_width, self.target_height))

                # Run YOLO detection
                results = self.model(frame, 
                                   classes=[0],
                                   conf=self.min_detection_confidence,
                                   iou=0.45,
                                   max_det=20)

                # Process detections
                current_centroids = []
                
                for det in results[0].boxes.data:
                    x1, y1, x2, y2, conf, cls = det
                    if conf < self.min_detection_confidence:
                        continue
                        
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                    current_centroids.append((centroid, (x1, y1, x2, y2)))

                # Update tracking with thread safety
                with self.lock:
                    if len(current_centroids) == 0:
                        for obj_id in tracked_objects:
                            tracked_objects[obj_id]['disappeared'] += 1
                    else:
                        # If we have no existing tracks, create new ones
                        if len(tracked_objects) == 0:
                            for centroid, bbox in current_centroids:
                                tracked_objects[next_object_id] = {
                                    'centroid': centroid,
                                    'frames': 1,
                                    'disappeared': 0,
                                    'bbox': bbox
                                }
                                next_object_id += 1
                        else:
                            # Match existing tracks to new detections
                            used_centroids = set()
                            for obj_id in list(tracked_objects.keys()):
                                if tracked_objects[obj_id]['disappeared'] > self.max_disappeared:
                                    del tracked_objects[obj_id]
                                    continue

                                obj_centroid = tracked_objects[obj_id]['centroid']
                                min_dist = float('inf')
                                closest_centroid = None
                                closest_bbox = None

                                for i, (centroid, bbox) in enumerate(current_centroids):
                                    if i in used_centroids:
                                        continue

                                    dist = np.sqrt((obj_centroid[0] - centroid[0])**2 + 
                                                 (obj_centroid[1] - centroid[1])**2)
                                    if dist < min_dist and dist < self.min_distance:
                                        min_dist = dist
                                        closest_centroid = centroid
                                        closest_bbox = bbox

                                if closest_centroid is not None:
                                    tracked_objects[obj_id]['centroid'] = closest_centroid
                                    tracked_objects[obj_id]['bbox'] = closest_bbox
                                    tracked_objects[obj_id]['frames'] += 1
                                    tracked_objects[obj_id]['disappeared'] = 0
                                    used_centroids.add(current_centroids.index((closest_centroid, closest_bbox)))

                                    # Check if this track should be confirmed
                                    if tracked_objects[obj_id]['frames'] >= self.min_frames and obj_id not in confirmed_ids:
                                        confirmed_ids.add(obj_id)
                                        self.total_cumulative_count += 1

                            # Create new tracks for unmatched detections
                            for i, (centroid, bbox) in enumerate(current_centroids):
                                if i not in used_centroids:
                                    tracked_objects[next_object_id] = {
                                        'centroid': centroid,
                                        'frames': 1,
                                        'disappeared': 0,
                                        'bbox': bbox
                                    }
                                    next_object_id += 1

                        # Update counts safely
                        self.current_count = sum(1 for obj_id in tracked_objects 
                                               if obj_id in confirmed_ids and 
                                               tracked_objects[obj_id]['disappeared'] == 0)
                        
                        # Update total count when new IDs are confirmed
                        for obj_id in tracked_objects:
                            if (tracked_objects[obj_id]['frames'] >= self.min_frames and 
                                obj_id not in confirmed_ids):
                                confirmed_ids.add(obj_id)
                                self.total_cumulative_count += 1

                # Visualization
                if not self.should_stop:
                    current_time = time.time()
                    for obj_id, obj_data in tracked_objects.items():
                        if obj_data['disappeared'] > 0:
                            continue
                            
                        x1, y1, x2, y2 = obj_data['bbox']
                        color = (0, 255, 0) if obj_id in confirmed_ids else (0, 165, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        if frame is not None:
                            last_frame_path = os.path.join(self.storage_path, "lastframe.png")
                            cv2.imwrite(last_frame_path, frame)
                            print(f"Last frame saved to {last_frame_path}")

                        if obj_id in confirmed_ids:
                            heatmap_data.append(obj_data['centroid'])
                            print(f"Added centroid to heatmap_data: {obj_data['centroid']}")
                            self.generate_heatmap(heatmap_data, frame.shape[1], frame.shape[0])
                    # Display counts
                    if current_time - last_time >= 1:
                        cv2.putText(frame, f'Current Count: {self.current_count}', (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f'Total Count: {self.total_cumulative_count}', (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        last_time = current_time
                    if heatmap_data and current_time - self.last_heatmap_update >= self.heatmap_update_interval:
                        print("Generating heatmap with the following data:", heatmap_data)
                        self.generate_heatmap(heatmap_data, frame.shape[1], frame.shape[0])
                        heatmap_image = cv2.imread(self.storage_path + "latest_heatmap.png")
                        if heatmap_image is not None:
                            heatmap_image = cv2.resize(heatmap_image, (frame.shape[1], frame.shape[0]))
                            frame = cv2.addWeighted(frame, 0.6, heatmap_image, 0.4, 0)
                        self.last_heatmap_update = current_time

                    cv2.imshow('Footfall Detection', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.processing_complete = True
            # Save the last frame as lastframe.png
            if frame is not None:
                last_frame_path = os.path.join(self.storage_path, "lastframe.png")
                cv2.imwrite(last_frame_path, frame)
                print(f"Last frame saved to {last_frame_path}")
            return self._get_current_stats()

    def _get_current_stats(self):
        with self.lock:
            # Generate heatmap for each zone
            heatmap_generator = HeatmapGenerator()
            heatmap_urls = []

            for zone_id, data in self.zone_data.items():
                if data['points']:
                    heatmap_url = heatmap_generator.generate_for_zone(
                        data['points'],
                        self.target_width,
                        self.target_height,
                        zone_id
                    )
                    print(f"Generated heatmap URL: {heatmap_url}")
                    heatmap_urls.append(heatmap_url)

            # Create response
            response = {
                'footfall_summary': {
                    'total_footfall': self.total_cumulative_count,
                    'zone_footfall': {
                        zone_id: data['current_count'] 
                        for zone_id, data in self.zone_data.items()
                    },
                    'high_density_times': []
                },
                'heatmap_urls': heatmap_urls,
                'processing_complete': self.processing_complete
            }

            # Add density analysis if processing is complete
            if self.processing_complete:
                density_analyzer = DensityAnalyzer()
                density_data = density_analyzer.analyze({
                    'total_count': self.total_cumulative_count,
                    'zone_counts': {
                        zone_id: data['current_count']
                        for zone_id, data in self.zone_data.items()
                    }
                })
                response['footfall_summary']['high_density_times'] = density_data['high_density_periods']

            return response

    def generate_heatmap(self, heatmap_data, width, height):
        print(f"Storage path for heatmap: {self.storage_path}")
        
        if not heatmap_data:
            print("No heatmap data to generate heatmap.")
            return

        # Create a blank heatmap
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Populate the heatmap with detection points
        for x, y in heatmap_data:
            if isinstance(x, int) and isinstance(y, int) and 0 <= x < width and 0 <= y < height:
                heatmap[y, x] += 1  # Increment the heatmap at the detection point
            else:
                print(f"Invalid centroid: ({x}, {y})")

        # Normalize the heatmap
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = np.uint8(heatmap)
        
        # Create a color map
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Check the shape of the heatmap color
        print(f"Heatmap color shape: {heatmap_color.shape}")
        
        # Save the heatmap
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{self.storage_path}latest_heatmap.png"
        
        try:
            cv2.imwrite(filepath, heatmap_color)
            print(f"Heatmap saved to {filepath}")
        except Exception as e:
            print(f"Error saving heatmap: {e}")

    def stop_processing(self):
        self.should_stop = True
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=3)
        
        self.processing_complete = True
        return self._get_current_stats()

    def _process_zones(self, frame, detections, zones):
        frame_height, frame_width = frame.shape[:2]
        
        # Initialize zones if not already done
        if not self.zones and zones:
            for zone in zones:
                zone_id = zone['zone_id']
                # Convert percentages to pixel coordinates
                x1 = int(zone['bounds']['x1'] * frame_width)
                y1 = int(zone['bounds']['y1'] * frame_height)
                x2 = int(zone['bounds']['x2'] * frame_width)
                y2 = int(zone['bounds']['y2'] * frame_height)
                self.zones[zone_id] = (x1, y1, x2, y2)
                self.zone_data[zone_id] = {
                    'current_count': 0,
                    'total_count': 0,
                    'points': []
                }

        # Reset current counts
        for zone_id in self.zones:
            self.zone_data[zone_id]['current_count'] = 0

        # Create a 4x4 grid
        grid_counts = np.zeros((4, 4), dtype=int)

        # Process each detection
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Determine which grid box the centroid falls into
            grid_x = min(3, center_x * 4 // frame_width)
            grid_y = min(3, center_y * 4 // frame_height)
            grid_counts[grid_y, grid_x] += 1

            # Check each zone
            for zone_id, (zx1, zy1, zx2, zy2) in self.zones.items():
                if (zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2):
                    self.zone_data[zone_id]['current_count'] += 1
                    self.zone_data[zone_id]['points'].append((center_x, center_y))

        # Find the grid box with the highest footfall
        highest_footfall_box = np.unravel_index(np.argmax(grid_counts, axis=None), grid_counts.shape)
        print(f"Highest footfall in box: {highest_footfall_box} with count: {grid_counts[highest_footfall_box]}")

        # Draw grid on the frame
        for i in range(1, 4):
            cv2.line(frame, (0, i * frame_height // 4), (frame_width, i * frame_height // 4), (255, 0, 0), 2)
            cv2.line(frame, (i * frame_width // 4, 0), (i * frame_width // 4, frame_height), (255, 0, 0), 2)

        return frame

    def update_footfall_per_second(self):
        current_time = datetime.now()
        if len(self.footfall_per_second) < 60:  # Keep track for the last 60 seconds
            self.footfall_per_second.append((current_time, self.current_count))
        else:
            self.footfall_per_second.pop(0)
            self.footfall_per_second.append((current_time, self.current_count))
        
        # Reset the active footfall count for the next second
        self.active_footfall_count = 0

    def generate_footfall_graph(self):
        times, counts = zip(*self.footfall_per_second)
        
        plt.figure(figsize=(10, 5))
        plt.plot(times, counts, marker='o')
        plt.title('Footfall per Second')
        plt.xlabel('Time')
        plt.ylabel('Footfall Count')
        
        # Format the x-axis to show time in seconds
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=1))  # Set major ticks to every second
        plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
        
        plt.tight_layout()
        plt.savefig('static/footfall_graph.png')
