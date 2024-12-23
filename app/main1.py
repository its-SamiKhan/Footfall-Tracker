from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Import CORS
from .video_processor import VideoProcessor
from .density_analyzer import DensityAnalyzer
from .heatmap_generator import HeatmapGenerator
import cv2
import numpy as np
from datetime import datetime
import os

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Serve the index.html file
@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/status')
def get_status():
    try:
        if hasattr(app, 'video_processor'):
            app.video_processor.update_footfall_per_second()  # Update footfall counts per second
            app.video_processor.generate_footfall_graph()  # Generate graph
            return jsonify({
                **app.video_processor._get_current_stats(),
                'footfall_graph_url': '/static/footfall_graph.png'
            })
        return jsonify({
            'footfall_summary': {
                'total_footfall': 0,
                'zone_footfall': {},
                'high_density_times': []
            },
            'heatmap_urls': [],
            'footfall_graph_url': None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        data = request.get_json()
        
        if not data.get('video_stream_url'):
            return jsonify({'error': 'video_stream_url is required'}), 400
            
        video_processor = VideoProcessor()
        app.video_processor = video_processor
        
        processing_thread = video_processor.process_stream(
            data['video_stream_url'],
            data.get('zones', [])
        )
        
        return jsonify({
            'status': 'processing_started',
            'footfall_summary': {
                'total_footfall': 0,
                'zone_footfall': {},
                'high_density_times': []
            },
            'heatmap_urls': []
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop', methods=['POST'])
def stop_processing():
    try:
        if hasattr(app, 'video_processor'):
            result = app.video_processor.stop_processing()
            delattr(app, 'video_processor')
            return jsonify(result)
        return jsonify({
            'error': 'No active processing found',
            'current_count': 0,
            'total_count': 0,
            'zone_counts': {}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 