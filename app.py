from flask import Flask, request, render_template
import google.generativeai as gen_ai
from face_emotion import emotions_extract
from image_to_ import detect_objects_and_scene
from scene_detect import classify_scene
import base64
from io import BytesIO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Set up the Google API Key and configure the generative AI model
GOOGLE_API_KEY = "AIzaSyA2JNXTR65XZ-wU3tVPcgoeW_bwKZagMPk"
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-pro')
def generate_story(emotions, objects, scene,customization):
    # Create a prompt based on the detected inputs  
    prompt = (
    f"Write a delightful and simple story for children based on the following details:\n\n"
    f"- Detected Emotions: {', '.join(emotions)}\n"
    f"- Detected Objects: {', '.join(objects)}\n"
    f"- Scene/Background: {scene}\n\n"
    "Please follow these guidelines:\n"
    f"{customization}"  
    )

    response = model.generate_content(prompt)
    return response.text.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-story', methods=['POST'])
def generate_story_from_image():
    if 'image' not in request.files:
        return "No file part", 400
    image = request.files['image']
    if image.filename == '':
        return "No selected file", 400

    # Convert the image to Base64
    image_data = BytesIO()
    img = Image.open(image)
    img.save(image_data, format=img.format)
    image_data.seek(0)
    encoded_image = base64.b64encode(image_data.read()).decode('utf-8')
    mime_type = f"image/{img.format.lower()}"  # Determine the MIME type (e.g., "image/jpeg", "image/png")
    image_path= f"data:{mime_type};base64,{encoded_image}"

    # Extract emotions, objects, and scenes from the uploaded image
    detected_emotions = emotions_extract(image_path)
    detected_objects = detect_objects_and_scene(image_path)
    detected_scene = classify_scene(image_path)

    customization = request.form.get('prompt')

    # Generate the story based on the extracted information
    story = generate_story(detected_emotions, detected_objects, detected_scene,customization)

    # Pass the Base64 image and story to the template
    return render_template("index.html", story=story, image_data=f"data:{mime_type};base64,{encoded_image}")

if __name__ == "__main__":
    app.run(debug=True)