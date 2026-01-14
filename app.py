import streamlit as st
import os
import tempfile
import cv2
import numpy as np

from model.predict import predict_disease
from utils.disease_info import DISEASE_INFO
from utils.weather import get_weather


# CONFIG

st.set_page_config(
    page_title="Agro-Scan | Plant Disease Detection",
    layout="centered"
)

VIRAL_KEYWORDS = ["virus", "mosaic", "curl"]
FUNGAL_KEYWORDS = ["blight", "rust", "mold", "spot"]
BACTERIAL_KEYWORDS = ["bacterial"]


# RESIZE WITH PADDING (YOUR FUNCTION)

def resize_with_padding(img, target_size=224):
    h, w, _ = img.shape

    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = target_size - new_w
    pad_h = target_size - new_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        resized,
        top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    return padded


# SEVERITY ESTIMATION

def estimate_severity(disease_name, confidence):
    disease = disease_name.lower()

    if "healthy" in disease:
        return "None", "🟢 Plant appears healthy."

    if confidence < 0.60:
        return "Unknown", "⚠️ Low confidence. Please retake the image."

    if any(k in disease for k in VIRAL_KEYWORDS):
        return "Severe", "🔴 Viral diseases spread rapidly and have no chemical cure."

    if any(k in disease for k in FUNGAL_KEYWORDS):
        if confidence > 0.85:
            return "Severe", "🔴 Advanced fungal infection likely."
        elif confidence > 0.70:
            return "Moderate", "🟠 Infection present but manageable."
        else:
            return "Mild", "🟢 Early-stage fungal infection."

    if confidence > 0.80:
        return "Moderate", "🟠 Disease present. Monitor closely."

    return "Mild", "🟢 Early symptoms detected."


# WEATHER RISK LOGIC

def weather_risk_advice(disease_name, weather_info):
    disease = disease_name.lower()
    advice = []

    try:
        temp = float(weather_info.split("Temperature:")[1].split("°")[0].strip())
        humidity = int(weather_info.split("Humidity:")[1].split("%")[0].strip())
    except Exception:
        return []

    if any(k in disease for k in FUNGAL_KEYWORDS) and humidity > 70:
        advice.append(
            "⚠️ High humidity detected — fungal disease may spread rapidly.\n"
            "- Improve air circulation\n"
            "- Avoid overhead irrigation"
        )

    if any(k in disease for k in BACTERIAL_KEYWORDS) and humidity > 65 and temp > 25:
        advice.append(
            "⚠️ Warm & humid conditions favor bacterial spread.\n"
            "- Sanitize tools\n"
            "- Avoid touching wet plants"
        )

    if any(k in disease for k in VIRAL_KEYWORDS) and temp > 25:
        advice.append(
            "⚠️ Warm weather increases insect vectors for viral diseases.\n"
            "- Monitor whiteflies/aphids\n"
            "- Remove infected plants early"
        )

    return advice


# UI

st.title("🌿 Agro-Scan – AI Plant Disease Detector")
st.write("Upload a plant leaf image to detect crop, disease, severity, and risk.")

uploaded_file = st.file_uploader(
    "📷 Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

city = st.text_input(
    "🌦️ Enter your city (optional – for weather-based insights)"
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    st.image(image_path, caption="Uploaded Image", use_container_width=True)

    
    # PREPROCESS IMAGE (NEW)
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_with_padding(img, target_size=224)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    
    # PREDICTION
    
    with st.spinner("🔍 Analyzing image..."):
        results = predict_disease(img, top_k=3)

    st.subheader("🔍 Top Predictions")

    for i, (disease_class, confidence) in enumerate(results, start=1):
        crop, disease = disease_class.split("___", 1)
        st.write(
            f"**{i}. 🌱 {crop} – 🦠 {disease.replace('_', ' ')}** "
            f"({confidence * 100:.2f}%)"
        )

    best_class, best_confidence = results[0]
    crop, disease = best_class.split("___", 1)

    
    # CONFIDENCE CHECK
    
    if best_confidence < 0.60:
        st.error(
            "❗ Low confidence prediction.\n\n"
            "Please retake the photo with:\n"
            "- Clear lighting\n"
            "- Single leaf\n"
            "- Minimal background"
        )
        os.remove(image_path)
        st.stop()

    
    # WEATHER INSIGHTS
    
    if city:
        try:
            weather_info = get_weather(city)
            st.subheader("🌦️ Local Weather Conditions")
            st.write(weather_info)

            risks = weather_risk_advice(disease, weather_info)
            if risks:
                st.subheader("⚠️ Weather-Based Disease Risk")
                for r in risks:
                    st.warning(r)
            else:
                st.info("✅ Weather conditions are not highly favorable for disease spread.")

        except Exception:
            st.warning("⚠️ Could not fetch weather data.")

    
    # SEVERITY
    
    severity, severity_msg = estimate_severity(disease, best_confidence)

    st.subheader("📊 Disease Severity Assessment")

    if severity == "Severe":
        st.error(f"Severity: 🔴 **{severity}**")
    elif severity == "Moderate":
        st.warning(f"Severity: 🟠 **{severity}**")
    else:
        st.success(f"Severity: 🟢 **{severity}**")

    st.write(severity_msg)

    
    # TREATMENT
    
    st.subheader("💊 Recommended Action")
    st.write(
        DISEASE_INFO.get(
            best_class,
            "ℹ️ No specific treatment information available."
        )
    )

    os.remove(image_path)

else:
    st.info("⬆️ Please upload a plant leaf image to begin.")
