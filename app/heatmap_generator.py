import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os

class HeatmapGenerator:
    def __init__(self):
        self.storage_path = "static/heatmaps/"
        # Ensure directory exists
        os.makedirs(self.storage_path, exist_ok=True)
        
    def generate_for_zone(self, points, width, height, zone_id):
        # Create heatmap data
        heatmap = np.zeros((height, width))
        
        # Add points to heatmap
        for x, y in points:
            if 0 <= x < width and 0 <= y < height:
                # Add Gaussian distribution around each point
                y_grid, x_grid = np.ogrid[-y:height-y, -x:width-x]
                gaussian = np.exp(-(x_grid*x_grid + y_grid*y_grid) / (2*30*30))
                heatmap += gaussian

        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
        
        # Generate heatmap visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap, cmap='hot')
        plt.colorbar()
        plt.title(f'Heatmap for Zone {zone_id}')
        
        # Save the heatmap
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"heatmap_zone_{zone_id}_{timestamp}.png"
        filepath = os.path.join(self.storage_path, filename)
        plt.savefig(filepath)
        plt.close()
        
        return f"/static/heatmaps/{filename}" 