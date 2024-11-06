from PIL import Image
import numpy as np
from transformers import BlipProcessor, TFBlipForConditionalGeneration
from config import IMAGE_SIZE

class ImageCaptioner:
    """
    A class to handle image captioning using a BLIP model.
    """

    # Load the BLIP processor and model once to avoid reloading for each request
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    @staticmethod
    def preprocess_image(image_path):
        """
        Opens and preprocesses an image for caption generation.

        Args:
            image_path (str): The path to the image file.
            
        Returns:
            PIL.Image: The preprocessed image ready for input to the model.
        
        Raises:
            ValueError: If an error occurs while processing the image.
        """
        try:
            # Open the image and ensure it is in RGB format
            img = Image.open(image_path).convert("RGB")
            
            # Resize the image to the standard size specified in config
            img = img.resize(IMAGE_SIZE)

            # Convert image to a numpy array and normalize pixel values to [0, 1]
            img_array = np.array(img) / 255.0
            
            # Convert back to PIL Image after normalization
            img = Image.fromarray((img_array * 255).astype(np.uint8))

            return img
        except Exception as e:
            raise ValueError(f"Error processing image: {e}")

    @staticmethod
    def generate_caption(image_path):
        """
        Generates a descriptive caption for a given image.
        
        Args:
            image_path (str): The path to the image file.
            
        Returns:
            str: The generated caption for the image.
        """
        # Preprocess the image
        img = ImageCaptioner.preprocess_image(image_path)

        # Prepare the image for model input
        inputs = ImageCaptioner.processor(images=img, return_tensors="tf")

        # Generate the caption using the BLIP model
        out = ImageCaptioner.model.generate(**inputs)
        caption = ImageCaptioner.processor.decode(out[0], skip_special_tokens=True)
        return caption
