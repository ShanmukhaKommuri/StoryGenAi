# StoryGenAi

This project is a web application that generates a creative story for children based on an uploaded image. The application uses Python back-end to process the image and generate the story.

---

## Features

- **Upload Image**: Users can upload an image through the web interface.
- **Image Processing**: The Python server analyzes the uploaded image to detect:
  - Emotions from faces
  - Objects in the image
  - Scene classification
- **AI-Powered Story Generation**: Uses Google Generative AI to create a simple and fun story based on the detected details.
- **Interactive Front-End**: Clean and user-friendly interface to upload images and view the generated story.

---

## Tech Stack

### Front-End
- **HTML/CSS/JavaScript**: For the user interface.

### Back-End
- **Python**: Used for image processing and story generation.
  - **Flask**: Lightweight web framework for creating REST APIs.
  - **Custom Modules**: Handles emotions detection, object detection, and scene classification.
- **Google Generative AI**: Generates creative stories based on image details.

---

## Installation

### Prerequisites
- **Python** (v3.9 or above)
- **Pip** for Python dependencies
- **npm** for Node.js dependencies

### Clone the Repository
```bash
git clone https://github.com/shanmukhaKommuri/StoryGenAi.git
cd StoryGenAi
