from app.main1 import app
import os

# Create static/heatmaps directory if it doesn't exist
os.makedirs('static/heatmaps', exist_ok=True)

if __name__ == '__main__':
    # Enable debug mode for development
    app.debug = True
    # Ensure static files are served properly
    app.static_folder = os.path.abspath('static')
    app.static_url_path = '/static'
    app.run(host='0.0.0.0', port=5000)