import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
import numpy as np
import cv2

model = load_model("C:\\Users\\rohith\\Downloads\my_model (2).h5")
# model.complile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

def preprocess_image(image):
    image = cv2.resize(image, (28, 28))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = np.expand_dims(image, axis=(0, -1))
    return image

st.title("Real-Time Letter Detection")

canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)

if "current_word" not in st.session_state:
    st.session_state['current_word'] = ""

if canvas_result.image_data is not None:
    image_data = canvas_result.image_data.astype("uint8")
    print(image_data.shape)
    processed_image = preprocess_image(image_data)
    prediction = model.predict(processed_image)
    detected_letter = chr(np.argmax(prediction) + 65)
    st.write(f"Detected Letter: {detected_letter}")
    if st.button("Add Letter"):
        st.session_state.current_word += detected_letter

st.write(f"Forming Word: {st.session_state.current_word}")