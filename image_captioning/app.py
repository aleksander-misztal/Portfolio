from flask import Flask, request, jsonify
import os
import uuid
import time
import requests
import logging
from werkzeug.utils import secure_filename
from ImageCaptioner import ImageCaptioner  # Import the ImageCaptioner class
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS

# Disable GPU usage in TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Show only warnings and errors
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

def allowed_file(filename):
    """
    Check if the uploaded file has a valid extension.

    Args:
        filename (str): The name of the file to check.

    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return {'res': 'test'}

@app.route('/generate_caption', methods=['POST'])
def generate_caption_api():
    """
    API endpoint to generate a caption for the image from the provided URL.

    Returns:
        JSON response containing the generated caption and filename,
        or an error message with appropriate HTTP status codes.
    """
    data = request.get_json()
    image_url = data.get('image_url')

    logging.info(f"Received request with image_url: {image_url}")

    if not image_url:
        logging.error("No image URL provided")
        return jsonify({"error": "No image URL provided"}), 400

    if not isinstance(image_url, str) or not image_url.startswith(('http://', 'https://')):
        logging.error("Invalid URL format")
        return jsonify({"error": "Invalid URL format"}), 400

    timestamp = int(time.time())
    unique_filename = f"{timestamp}_{uuid.uuid4()}.jpg"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    try:
        logging.info(f"Downloading image from {image_url}...")
        response = requests.get(image_url)
        response.raise_for_status()

        with open(image_path, 'wb') as file:
            file.write(response.content)

        logging.info(f"Image downloaded and saved as {unique_filename}")

        caption = ImageCaptioner.generate_caption(image_path)

        logging.info(f"Caption generated for {unique_filename}")
        return jsonify({'caption': caption, 'filename': unique_filename}), 200

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading image: {str(e)}")
        return jsonify({"error": f"Error downloading image: {str(e)}"}), 500

    except Exception as e:
        logging.error(f"Error processing the image: {str(e)}")
        return jsonify({"error": str(e)}), 500