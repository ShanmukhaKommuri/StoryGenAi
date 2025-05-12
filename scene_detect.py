# import torch
# from torchvision import models, transforms
# from PIL import Image
# import os
# import base64
# from io import BytesIO

# def load_places365_model():
#     model = models.resnet50(num_classes=365)  # 365 classes for Places365
#     checkpoint = torch.hub.load_state_dict_from_url(
#         "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar",
#         map_location=torch.device("cpu")
#     )
#     state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
#     model.load_state_dict(state_dict)
#     model.eval()
#     return model

# def classify_scene(image_path):
#     base64_data = image_path.split(",")[1]
    
#     # Decode it to an image
#     image_data = base64.b64decode(base64_data)
#     image = Image.open(BytesIO(image_data))
#     model = load_places365_model()

    
#     preprocess = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     image = Image.open(image).convert("RGB")
#     input_tensor = preprocess(image).unsqueeze(0)

    
#     with torch.no_grad():
#         output = model(input_tensor)
#         probabilities = torch.nn.functional.softmax(output[0], dim=0)

    
#     classes_file = "categories_places365.txt"
#     if not os.path.exists(classes_file):
#         classes_file_url = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
#         torch.hub.download_url_to_file(classes_file_url, classes_file)
#     with open(classes_file) as f:
#         categories = [line.strip().split(" ")[0][3:] for line in f.readlines()]

#     # Get the top-1 prediction (scene with highest probability)
#     top1_prob, top1_catid = torch.max(probabilities, dim=0)
#     print(f"The scene is '{categories[top1_catid]}' with a probability of {top1_prob.item():.4f}")

# # # Example usage
# # image_path = "utils/image_2.png"
# # classify_scene(image_path)


# # # Example usage
# # image_path = "utils/image_2.png"
# # classify_scene(image_path)
import torch
from torchvision import models, transforms
from PIL import Image
import os
import base64
from io import BytesIO

def load_places365_model():
    model = models.resnet50(num_classes=365)  # 365 classes for Places365
    checkpoint = torch.hub.load_state_dict_from_url(
        "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar",
        map_location=torch.device("cpu")
    )
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

def classify_scene(image_path):
    try:
        base64_data = image_path.split(",")[1]
        
        # Decode base64 image to raw image data
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data)).convert("RGB")  # Ensure RGB format

        # Load the pre-trained model
        model = load_places365_model()

        # Define image preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the required size for ResNet50
            transforms.ToTensor(),          # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])

        # Preprocess the image
        input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Run the model to get the output
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Get probabilities

        # Load the category names for Places365
        classes_file = "categories_places365.txt"
        if not os.path.exists(classes_file):
            classes_file_url = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
            torch.hub.download_url_to_file(classes_file_url, classes_file)

        with open(classes_file) as f:
            categories = [line.strip().split(" ")[0][3:] for line in f.readlines()]  # Clean the category names

        # Get the top-1 prediction (scene with highest probability)
        top1_prob, top1_catid = torch.max(probabilities, dim=0)
        print(f"The scene is '{categories[top1_catid]}' with a probability of {top1_prob.item():.4f}")

    except Exception as e:
        print(f"Error during scene classification: {e}")

# Example usage (you would pass a base64-encoded image string here)
# image_path = "<base64_encoded_image_string>"
# classify_scene(image_path)
