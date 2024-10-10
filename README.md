# VisionAI: Real-Time Multimodal Assistant with Generative Intelligence


## Description

VisionAI is an advanced real-time multimodal assistant that leverages state-of-the-art AI technologies. It can detect and classify objects in images using YOLOv8, generate captions using a Vision Transformer (ViT) model, generate images from text prompts with Stable Diffusion, and transcribe audio to text. Additionally, it integrates chatbot functionality using the LLaMA model to provide intelligent responses based on text and image context.


### Features

-Object Detection: Real-time object detection and classification using the YOLOv8 model.
-Image Captioning: Generate descriptive captions for uploaded images using a Vision Transformer and GPT-2 model.
-Chatbot with Image Context: Chatbot responds to user queries with image context using a pre-trained LLaMA model.
-Text-to-Image Generation: Generate high-quality images from uncensored text prompts using Stable Diffusion.
-Speech Recognition: Upload audio files to transcribe spoken words into text using Google Speech Recognition.

### Demo

# Functionalities
1. Object Detection
Upload an image, and the system will detect objects and display bounding boxes with classification labels.

2. Image Captioning
Upload an image, and VisionAI will generate a descriptive caption using the ViT-GPT2 model.

3. Chatbot Response with Image Context
Upload an image, ask a question related to it, and VisionAI will provide a relevant response considering both the image context and the user's query.

4. Image Generation
Input an uncensored text prompt to generate high-quality images using Stable Diffusion.


# Tech Stack

- Streamlit: The framework used for creating the user interface.
- YOLOv8: Object detection and classification model for real-time object recognition.
- VisionEncoderDecoderModel: Used for image captioning (ViT + GPT-2).
- Ollama (LLaMA): For chatbot responses with image-based query answering.
- Stable Diffusion: Used to generate images based on uncensored text prompts.
- SpeechRecognition: To convert audio to text.
- OpenCV: Used for image processing and object detection display.


# Installation
1. Clone the repository:

git clone https://github.com/your-username/VisionAI.git
cd VisionAI

2. Create a virtual environment and activate it:

python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install the required dependencies:

pip install -r requirements.txt

4. Download the YOLOv8 pre-trained weights:

wget https://github.com/ultralytics/yolov8/releases/download/v8.0/yolov8n.pt

5. Usage
- Run the Streamlit app:

streamlit run app.py

- Open your web browser and navigate to the provided URL (usually http://localhost:8501).

# File Structure

VisionAI/
│ 
├── app.py                  # Main Streamlit app
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── yolov8n.pt              # YOLOv8 pre-trained weights (downloaded externally)


# Acknowledgements

YOLOv8
Stable Diffusion
Hugging Face Transformers
Streamlit
Google Speech Recognition

