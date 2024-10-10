import streamlit as st
import cv2
import numpy as np
from PIL import Image
import speech_recognition as sr
from ultralytics import YOLO
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import torch

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Streamlit app title
st.title('VisionAI: Real-Time Multimodal Assistant with Generative Intelligence')

# Load image captioning model
@st.cache_resource
def load_image_captioning_model():
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return processor, model, tokenizer

def caption_image(image):
    processor, model, tokenizer = load_image_captioning_model()
    pixel_values = processor(images=[image], return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Function to detect and classify objects using YOLOv8
def detect_objects_yolov8(image):
    image_cv = np.array(image)
    results = model(image_cv)
    boxes, confidences, class_ids = [], [], []
    for result in results:
        for det in result.boxes.data:
            x1, y1, x2, y2, conf, class_id = det
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            confidences.append(float(conf))
            class_ids.append(int(class_id))
    return boxes, confidences, class_ids

# Load Stable Diffusion model for text-to-image generation
@st.cache_resource
def load_stable_diffusion_model():
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    pipe = pipe.to("cpu")  # Ensure it runs on CPU
    return pipe

# Function to generate an image based on a text prompt
def generate_image(prompt, stable_diffusion_model):
    # Generate an image from a text prompt using the Stable Diffusion model
    image = stable_diffusion_model(prompt).images[0]
    return image

# Input field for user query (text)
input_text = st.text_input("Ask something:")

# File uploader for images
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Process text and image together
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    task = st.radio("Choose a task", ("Object Detection", "Image Captioning", "Chatbot Response with Image Context", "Image Generation"))

    with st.spinner('Processing...'):
        if task == "Object Detection":
            boxes, confidences, class_ids = detect_objects_yolov8(image)
            image_cv = np.array(image)
            class_labels = model.names  # Preloaded COCO class labels

            for i, box in enumerate(boxes):
                x, y, w, h = box
                label = class_labels[class_ids[i]]
                confidence = confidences[i]
                cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label_text = f"{label}: {confidence:.2f}"
                cv2.putText(image_cv, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            st.image(image_cv, caption="Detected and Classified Objects", use_column_width=True)

        elif task == "Image Captioning":
            caption = caption_image(image)
            st.write(f"Caption: {caption}")

        elif task == "Chatbot Response with Image Context":
            caption = caption_image(image)
            st.write(f"Caption: {caption}")
            if input_text:
                template = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are a helpful assistant. You also process images."),
                        ("user", "The image contains: {caption}. Also, here is my question: {query}")
                    ]
                )
                llm = Ollama(model="llama3")
                llm_chain = LLMChain(prompt=template, llm=llm)
                response = llm_chain.invoke({"caption": caption, "query": input_text})
                st.write(f"Response: {response['text']}")

        elif task == "Image Generation":
            # Ask for uncensored prompt from the user
            uncensored_prompt = st.text_area("Enter your uncensored prompt for image generation:")
            
            if uncensored_prompt:
                # Load image generation model
                stable_diffusion_model = load_stable_diffusion_model()
                
                # Generate image based on the uncensored prompt
                generated_image = generate_image(uncensored_prompt, stable_diffusion_model)
                
                # Display the generated image
                st.image(generated_image, caption="Generated Image", use_column_width=True)

# File uploader for audio
audio_file = st.file_uploader("Upload an audio file for speech recognition", type=["wav", "mp3"])

# Process audio input
if audio_file is not None:
    recognizer = sr.Recognizer()
    audio_data = sr.AudioFile(audio_file)
    with audio_data as source:
        audio = recognizer.record(source)
    text_from_audio = recognizer.recognize_google(audio)
    st.write(f"Transcribed Text: {text_from_audio}")

# Process text input using LLaMA 2 from Ollama
if input_text and uploaded_file is None:
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please respond to the user's queries."),
            ("user", "Question: {query}")
        ]
    )
    
    llm = Ollama(model="llama3")
    llm_chain = LLMChain(prompt=template, llm=llm)
    response = llm_chain.invoke({"query": input_text})
    st.write(f"Response: {response['text']}")
