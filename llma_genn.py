from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def generate_story_with_llama(objects, emotions, scene):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")  # Replace with the actual path
    model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")

    # Create the story prompt
    prompt = (
        f"Write a vivid and engaging short story for children. "
        f"The story takes place in a {scene}, where the characters include {', '.join(objects)}. "
        f"The story should convey the emotions of {', '.join(emotions)}."
    )

    # Use pipeline for text generation
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    story = generator(prompt, max_length=300, num_return_sequences=1, temperature=0.7)

    return story[0]["generated_text"]

# Example usage
detected_objects = ["dog", "child", "bicycle"]
detected_emotions = ["happiness", "curiosity"]
detected_scene = "sunlit meadow"
story = generate_story_with_llama(detected_objects, detected_emotions, detected_scene)
print("Generated Story:", story)
