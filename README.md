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
![Screenshot 2024-10-10 163643](https://github.com/user-attachments/assets/719ac6c5-d1e3-4919-ba87-3e7436a06427)

![Screenshot 2024-10-10 164302](https://github.com/user-attachments/assets/d42abb49-2fdb-4cbb-9a1f-8778537d5a5b)

![Screenshot 2024-10-10 164325](https://github.com/user-attachments/assets/ecd1363e-8b6b-41a6-985f-093f0a82ad4f)


# Object Detection
Upload an image, and the system will detect objects and display bounding boxes with classification labels.
![Screenshot 2024-10-10 164723](https://github.com/user-attachments/assets/7cccc8b8-dd1a-49c2-a3ff-e2e5df175e2e)

# Image Captioning
Upload an image, and VisionAI will generate a descriptive caption using the ViT-GPT2 model.
![Screenshot 2024-10-10 164812](https://github.com/user-attachments/assets/b95fd339-62bb-4ee8-9ea7-722e4bcfeeb9)


# Chatbot Response with Image Context
Upload an image, ask a question related to it, and VisionAI will provide a relevant response considering both the image context and the user's query.
![Screenshot 2024-10-10 170246](https://github.com/user-attachments/assets/f0866885-9530-4c10-a41a-dea0c381854a)

# Image Generation
Input an uncensored text prompt to generate high-quality images using Stable Diffusion.
![Screenshot 2024-10-10 165814](https://github.com/user-attachments/assets/1fc38e39-1621-4867-b4f7-4d8f0fa6bfa9)





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
├── app.py                 
├── README.md               
├── requirements.txt        
└── yolov8n.pt               
(downloaded externally)


# Acknowledgements

YOLOv8
Stable Diffusion
Hugging Face Transformers
Streamlit
Google Speech Recognition

