from deepface import DeepFace
import base64
from io import BytesIO
from PIL import Image
import numpy

def emotions_extract(image_path):
    base64_data = image_path.split(",")[1]
    
    # Decode it to an image
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data))
    img_array = numpy.array(image)
    analysis = DeepFace.analyze(img_array, actions=['emotion'])

    print(f"Dominant Emotion: {analysis[0]['dominant_emotion']}")
    return analysis[0]['dominant_emotion']