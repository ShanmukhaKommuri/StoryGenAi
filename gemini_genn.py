# from dotenv import load_dotenv
import google.generativeai as gen_ai
from face_emotion import emotions_extract
from image_to_ import detect_objects_and_scene
from scene_detect import classify_scene

GOOGLE_API_KEY = "AIzaSyA2JNXTR65XZ-wU3tVPcgoeW_bwKZagMPk"

# Configure the Gemini-Pro AI model
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-pro')


def generate_story(emotions, objects, scene):
    # Create a prompt based on the detected inputs  
    prompt = (
    f"Write a simple story that should contain a moral for children based on the following:\n\n"
    f"- Detected Emotions: {', '.join(emotions)}\n"
    f"- Detected Objects: {', '.join(objects)}\n"
    f"- Scene/Background: {scene}\n\n"
    "The story should be easy to understand, with simple words. Use the emotions, objects, and scene to create a fun and happy story. "
    "Make sure the story is gentle and friendly for kids, with a happy ending."
    )
 

    response = model.generate_content(prompt)
    
    return response.text.strip()



if __name__ == "__main__":
    # Detected inputs
    
    # detected_emotions = ["happiness", "curiosity"]
    # detected_objects = ["dog", "ball", "tree"]
    # detected_scene = "park"
    image_path = 'utils/image.png'
    detected_emotions = emotions_extract(image_path)
    detected_objects = detect_objects_and_scene(image_path)
    detected_scene = classify_scene(image_path)
    
    # Generate the story
    story = generate_story(detected_emotions, detected_objects, detected_scene)
    
    # Print the story
    print("Generated Story:")
    print(story)
