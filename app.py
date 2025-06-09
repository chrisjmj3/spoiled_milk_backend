import cv2
import numpy as np
import io
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173","https://milk-spoilage-detection-kv3v.vercel.app/"])  # Change port if needed

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Reference RGB values (pure sample)
pure = [22.67081384872796, 60.17126431850358, 101.615427517268]

# Function to calculate ΔE (Euclidean distance in RGB space)
def delta_e(rgb1, rgb2):
    rgb1 = np.array(rgb1)
    rgb2 = np.array(rgb2)
    return np.sqrt(np.sum((rgb1 - rgb2) ** 2))

def classify_milk(delta_e_value):
    if delta_e_value < 30:
        return "fresh milk"
    elif 30 <= delta_e_value <= 60:
        return "spoiling"
    else:
        return "spoiled"


def estimate_ph(delta_e_value):
    # This is a simplified estimation - you might need a more accurate model
    # Assuming pH ranges from ~6.8 (fresh) to ~4.5 (spoiled)
    if delta_e_value < 30:
        return round(6.8 - (delta_e_value / 30) * 0.2, 1)  # Fresh: pH 6.8-6.6
    elif delta_e_value < 60:
        return round(6.6 - ((delta_e_value - 30) / 30) * 1.1, 1)  # Spoiling: pH 6.6-5.5
    else:
        return round(5.5 - min(1, (delta_e_value - 60) / 40), 1)  # Spoiled: pH 5.5-4.5

def process_image(image_data):
    # Convert image data to OpenCV format
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Check if image was loaded successfully
    if image is None:
        return {"error": "Failed to process the image."}
    
    # Convert to Lab color space for better color clustering
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    
    # Reshape for K-Means clustering
    pixels = lab.reshape((-1, 3))
    
    # Apply K-Means clustering (K=3 for background, strip, and others)
    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(np.float32(pixels), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Reshape labels back to image dimensions
    labels = labels.reshape(image.shape[:2])
    
    # Identify the strip cluster (lowest brightness in L-channel)
    strip_cluster = np.argmin(centers[:, 0])  # L-channel determines brightness
    
    # Create a binary mask for the strip
    strip_mask = (labels == strip_cluster).astype(np.uint8) * 255
    
    # Find contours of the strip
    contours, _ = cv2.findContours(strip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If a valid strip is found, process it
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)  # Get the largest contour (strip)
        x, y, w, h = cv2.boundingRect(largest_contour)  # Bounding box around the strip
        roi = image[y:y+h, x:x+w]  # Crop the strip region
        
        # Check if the ROI is valid
        if roi.size == 0:
            return {"error": "No valid region of interest found."}
        
        # Convert ROI to RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Compute the average RGB values of the strip
        average_rgb = np.mean(roi_rgb, axis=(0, 1))
        
        # Calculate ΔE
        delta_e_value = delta_e(average_rgb, pure)
        print(delta_e_value)
        
        # Classify milk freshness
        freshness_status = classify_milk(delta_e_value)
        
        # Encode the ROI image to return to frontend
        _, roi_encoded = cv2.imencode('.jpg', roi)
        roi_base64 = roi_encoded.tobytes()
        
        # Return the results
        return {
            "success": True,
            "rgb_values": average_rgb.tolist(),
            "delta_e": float(delta_e_value),
            "freshness": freshness_status,
            "roi_image": roi_base64
        }
    else:
        return {"error": "No strip detected. Adjust clustering parameters if needed."}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files.get('image')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Read the file
        image_data = file.read()
        
        # Process the image
        result = process_image(image_data)
        
        # Save the file for reference (optional)
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        # If there's an error, return it
        if "error" in result:
            return jsonify({"error": result["error"]}), 400

        # Estimate pH based on delta_e
        ph_value = estimate_ph(result.get("delta_e", 0))
        
        return jsonify({
            "result": result.get("freshness"),
            "delta_e": result.get("delta_e"),
            "ph": ph_value
        }), 200

if __name__ == '__main__':
    app.run(debug=True)
