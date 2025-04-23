# Image Captioning API

A Flask-based API that generates image captions using the BLIP model from Hugging Face.

## Features

- Accepts image URLs via POST request
- Downloads and preprocesses the image
- Uses a transformer model to generate captions
- Containerized with Docker

## Setup

### Docker

Build and start the application:

```bash
docker-compose up --build
```

The service will run at `http://localhost:5001`.

### Local

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask app:

```bash
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000
```

## API

### POST /generate_caption

**Request:**
```json
{
  "image_url": "https://example.com/image.jpg"
}
```

**Response:**
```json
{
  "caption": "Generated description of the image.",
  "filename": "timestamp_uuid.jpg"
}
```

## Configuration

Edit `config.py` to set:

- Image size
- Upload folder
- Allowed file extensions

## Notes

- TensorFlow GPU usage is disabled by default.
- Logs are written to `app.log`.
