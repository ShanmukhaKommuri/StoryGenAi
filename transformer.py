def generate_story(objects, emotions, scene):
    from transformers import pipeline

    story_generator = pipeline("text-generation", model="gpt2")
    prompt = (
        f"In a beautiful {scene}, there are {', '.join(objects)}. "
        f"The story follows their adventure where they experience emotions like {', '.join(emotions)}. "
        f"The characters face challenges, make friends, and learn valuable lessons about life. "
        f"Write a short, fun, and exciting story that captures the joy, adventure, and curiosity of these characters."
    )
    story = story_generator(prompt, max_new_tokens=250, num_return_sequences=1)
    return story[0]['generated_text']

# Example usage
detected_objects = ["playful dog", "curious child", "old bicycle"]
detected_emotions = ["joy", "adventure", "curiosity"]
detected_scene = "sunlit meadow with tall grass and a sparkling stream"

story = generate_story(detected_objects, detected_emotions, detected_scene)
print("Generated Story:", story)
