
import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from googletrans import Translator
import cv2
import base64
from io import BytesIO
from reportlab.pdfgen import canvas

# Load Model and Class Indices
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "Copy of Plant_Disease_Prediction_CNN_Image_Classifier.h5")
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(os.path.join(working_dir, "class_indices.json")))

# Precautions Dictionary with Attractive Bullets
precautions = {
    "Healthy": "✅ Your plant is healthy. Keep maintaining proper care. 🌱",
    "Apple___Apple_scab": "🔸 Use disease-resistant apple varieties. \n🔸 Prune trees to improve air circulation. \n🔸 Apply fungicides like captan or mancozeb.",
    "Apple___Black_rot": "🔹 Remove infected branches and fruit. \n🔹 Apply fungicides during bloom time. \n🔹 Avoid overhead watering to reduce moisture.",
    "Apple___Cedar_apple_rust": "🍎 Plant rust-resistant apple varieties. \n🍎 Remove nearby cedar trees if possible. \n🍎 Apply protective fungicides in early spring.",
    "Blueberry___healthy": "✅ Your blueberry plant is healthy! Keep maintaining good soil and watering habits. 🫐",
    "Cherry_(including_sour)___Powdery_mildew": "🍒 Ensure good air circulation. \n🍒 Use sulfur-based fungicides. \n🍒 Remove and dispose of infected leaves.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "🌽 Rotate crops yearly. \n🌽 Use disease-resistant corn hybrids. \n🌽 Apply fungicides containing strobilurins.",
    "Corn_(maize)___Common_rust_": "🌽 Grow rust-resistant corn varieties. \n🌽 Avoid overhead irrigation. \n🌽 Apply fungicides like mancozeb if necessary.",
    "Corn_(maize)___Northern_Leaf_Blight": "🌽 Rotate crops to prevent disease recurrence. \n🌽 Use resistant hybrids. \n🌽 Apply fungicides early in the season.",
    "Grape___Black_rot": "🍇 Remove mummified berries. \n🍇 Prune vines to increase air circulation. \n🍇 Apply protective fungicides in spring.",
    "Grape___Esca_(Black_Measles)": "🍇 Avoid pruning in wet conditions. \n🍇 Apply fungicides early in the growing season. \n🍇 Remove infected wood to prevent spreading.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "🍇 Maintain good vineyard sanitation. \n🍇 Prune to improve air circulation. \n🍇 Apply copper-based fungicides.",
    "Orange___Huanglongbing_(Citrus_greening)": "🍊 Remove and destroy infected trees. \n🍊 Control psyllid insect populations. \n🍊 Use disease-free nursery stock.",
    "Peach___Bacterial_spot": "🍑 Avoid overhead irrigation. \n🍑 Use copper-based sprays. \n🍑 Plant disease-resistant varieties.",
    "Pepper,_bell___Bacterial_spot": "🫑 Avoid working with wet plants. \n🫑 Rotate crops regularly. \n🫑 Use copper-based fungicides.",
    "Potato___Early_blight": "🥔 Space plants properly for air circulation. \n🥔 Apply fungicides like chlorothalonil. \n🥔 Remove infected leaves promptly.",
    "Potato___Late_blight": "🥔 Destroy infected plant debris. \n🥔 Use blight-resistant potato varieties. \n🥔 Apply systemic fungicides regularly.",
    "Raspberry___healthy": "✅ Your raspberry plant is healthy! Keep maintaining good care. 🍇",
    "Squash___Powdery_mildew": "🎃 Water plants at the base, not on leaves. \n🎃 Use sulfur or potassium bicarbonate sprays. \n🎃 Remove infected leaves immediately.",
    "Strawberry___Leaf_scorch": "🍓 Avoid overhead watering. \n🍓 Remove infected leaves. \n🍓 Apply fungicides early in the season.",
    "Tomato___Bacterial_spot": "🍅 Avoid handling wet plants. \n🍅 Use copper-based sprays. \n🍅 Rotate crops yearly to prevent reinfection.",
    "Tomato___Early_blight": "🍅 Provide good air circulation. \n🍅 Remove and destroy infected leaves. \n🍅 Apply fungicides like mancozeb or chlorothalonil.",
    "Tomato___Late_blight": "🍅 Use blight-resistant tomato varieties. \n🍅 Apply systemic fungicides. \n🍅 Destroy infected plants immediately.",
    "Tomato___Leaf_Mold": "🍅 Increase greenhouse ventilation. \n🍅 Reduce humidity by spacing plants apart. \n🍅 Apply copper-based fungicides.",
    "Tomato___Septoria_leaf_spot": "🍅 Remove and destroy infected leaves. \n🍅 Water at the base of the plant. \n🍅 Apply protective fungicides.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "🍅 Spray leaves with water to remove mites. \n🍅 Introduce natural predators like ladybugs. \n🍅 Use insecticidal soap or neem oil.",
    "Tomato___Target_Spot": "🍅 Apply copper-based fungicides. \n🍅 Rotate crops regularly. \n🍅 Keep foliage dry by watering at the base.",
    "Tomato___Yellow_Leaf_Curl_Virus": "🍅 Control whitefly populations. \n🍅 Use resistant tomato varieties. \n🍅 Remove and destroy infected plants.",
    "Tomato___Tomato_mosaic_virus": "🍅 Avoid tobacco use near tomato plants. \n🍅 Sanitize tools between uses. \n🍅 Remove infected plants immediately."
}

# Translator Object
translator = Translator()

# Function to Translate Text
def translate_text(text, target_language):
    if target_language == "English":
        return text
    return translator.translate(text, dest=target_language).text

# Function to Load & Preprocess Image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.  
    return img_array

# Function to Predict Image Class with Confidence Score
def predict_image_class(image_path):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions) * 100
    predicted_class = class_indices[str(predicted_class_index)]
    
    return predicted_class, confidence_score

# Rest of the Streamlit app remains the same
# ...
# Function to Capture Image from Webcam
def capture_image():
    camera = cv2.VideoCapture(0)
    st.write("Capturing Image...")
    ret, frame = camera.read()
    camera.release()
    
    if ret:
        img_path = "captured_image.jpg"
        cv2.imwrite(img_path, frame)
        return img_path
    return None

# Function to Export Report as PDF
def generate_pdf(prediction, confidence, precautions_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    
    c.drawString(100, 800, f"Plant Disease Prediction Report")
    c.drawString(100, 780, f"Prediction: {prediction}")
    c.drawString(100, 760, f"Confidence: {confidence:.2f}%")
    c.drawString(100, 740, f"Precautions: {precautions_text}")
    
    c.save()
    buffer.seek(0)
    return buffer

# Streamlit UI
st.sidebar.title("🌍 Language Selection")
languages = {
    "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr",
    "German": "de", "Tamil": "ta", "Telugu": "te", "Malayalam": "ml"
}
selected_language = st.sidebar.selectbox("Choose Language", list(languages.keys()))
lang_code = languages[selected_language]

st.sidebar.title(translate_text("Navigation", lang_code))
page = st.sidebar.radio(translate_text("Go to", lang_code), 
                        [translate_text("Home", lang_code), 
                         translate_text("About", lang_code), 
                         translate_text("FAQ", lang_code), 
                         translate_text("Feedback", lang_code)])

# Home Page
if page == translate_text("Home", lang_code):
    st.title(translate_text("🌿 Plant Disease Classifier", lang_code))
    st.write(translate_text("Upload an image or capture from camera.", lang_code))

    uploaded_image = st.file_uploader(translate_text("Upload an image...", lang_code), type=["jpg", "jpeg", "png"])
    captured_image_path = None

    if st.button("📷 Capture Image"):
        captured_image_path = capture_image()
    
    if uploaded_image or captured_image_path:
        image_path = uploaded_image if uploaded_image else captured_image_path
        image = Image.open(image_path)
        st.image(image, caption=translate_text("Uploaded Image", lang_code), use_column_width=True)

        if st.button(translate_text('Classify', lang_code)):
            prediction, confidence = predict_image_class(image_path)
            translated_prediction = translate_text(prediction, lang_code)
            confidence_text = f"{confidence:.2f}%"
            
            st.success(f'{translate_text("Prediction", lang_code)}: **{translated_prediction}**')
            st.info(f'{translate_text("Confidence", lang_code)}: {confidence_text}')
            
            # Display Precautionary Measures
            precaution_text = precautions.get(prediction, "No precautions available.")
            translated_precaution = translate_text(precaution_text, lang_code)
            st.warning(f'{translate_text("Precautions", lang_code)}: {translated_precaution}')

            # Export Report as PDF
            pdf_buffer = generate_pdf(prediction, confidence, precaution_text)
            b64_pdf = base64.b64encode(pdf_buffer.read()).decode("utf-8")
            href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="report.pdf">{translate_text("Download Report", lang_code)}</a>'
            st.markdown(href, unsafe_allow_html=True)

# About Page
elif page == translate_text("About", lang_code):
    st.title(translate_text("📖 About", lang_code))
    st.write(translate_text("""
    ### 🌱 About This Application  
    The **Plant Disease Classifier** is an AI-based tool that helps farmers detect diseases in plants using **deep learning**.  
      
    ### 🔍 How Does It Work?  
    - The model is trained on a dataset of plant diseases using **Convolutional Neural Networks (CNNs)**.  
    - Users upload an image of a plant leaf, and the model predicts the disease.  
      
    ### 🚀 Features  
    - **Live Camera Detection**  
    - **Multi-Language Support**  
    - **Confidence Score & Precautions**  
    - **Downloadable PDF Report**  
    """, lang_code))

# FAQ Page
elif page == translate_text("FAQ", lang_code):
    st.title(translate_text("❓ Frequently Asked Questions", lang_code))
    with st.expander(translate_text("📌 What type of images can I upload?", lang_code)):
        st.write(translate_text("You can upload clear images of plant leaves in JPG, JPEG, or PNG format.", lang_code))
    with st.expander(translate_text("📌 How accurate is the model?", lang_code)):
        st.write(translate_text("The model achieves 85-95% accuracy depending on image quality and dataset.", lang_code))
    with st.expander(translate_text("📌 Can this app suggest treatments?", lang_code)):
        st.write(translate_text("Currently, it only identifies diseases, but future updates will include treatment suggestions.", lang_code))
    with st.expander(translate_text("📌 Is this tool free to use?", lang_code)):
        st.write(translate_text("Yes! The app is completely free.", lang_code))

# Feedback Page
elif page == translate_text("Feedback", lang_code):
    st.title(translate_text("📝 Feedback", lang_code))
    name = st.text_input(translate_text("Enter your name", lang_code))
    feedback = st.text_area(translate_text("Your Feedback", lang_code))
    if st.button(translate_text("Submit", lang_code)):
        st.success(translate_text("Thank you for your valuable feedback!", lang_code))



