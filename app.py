import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from datetime import datetime

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="wide"
)

# --------------------------------
# CUSTOM DESIGN
# --------------------------------

st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: bold;
    color: #16834A;
    text-align: center;
}
.subtitle {
    text-align: center;
    font-size: 17px;
    color: gray;
}
.result-box {
    padding: 18px;
    border-radius: 12px;
    background-color: #e8f5e9;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# LOAD MODEL
# --------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/waste_classifier.keras"
    )

model = load_model()

with open("models/class_names.json", "r") as file:
    class_names = json.load(file)

# --------------------------------
# WASTE DISPOSAL ADVICE
# --------------------------------

advice = {
    "cardboard": "Keep cardboard clean and dry. Send it for paper recycling.",
    "glass": "Separate glass and send it to a glass recycling centre.",
    "metal": "Collect metal separately for metal recycling.",
    "paper": "Keep paper dry and place it in paper recycling.",
    "plastic": "Clean plastic containers and place them in plastic recycling.",
    "trash": "Dispose of mixed or other waste according to local waste rules."
}

# --------------------------------
# SESSION HISTORY
# --------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------
# HEADER
# --------------------------------

st.markdown(
    '<div class="main-title">♻️ Smart Waste Classification System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered waste identification and disposal guidance</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.title("♻️ Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Home", "Classification History", "About Project"]
)

# --------------------------------
# HOME PAGE
# --------------------------------

if page == "Home":

    st.header("Upload Waste Image")

    st.write(
        "Upload an image of plastic, paper, cardboard, "
        "glass, metal or general trash."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(
                image,
                caption="Your selected waste image",
                width="stretch"
            )

        with col2:
            st.subheader("Classification")

            if st.button("Classify Waste", type="primary"):

                with st.spinner("Analyzing image..."):

                    resized_image = image.resize((160, 160))

                    image_array = np.array(
                        resized_image,
                        dtype=np.float32
                    )

                    image_array = np.expand_dims(
                        image_array,
                        axis=0
                    )

                    predictions = model.predict(
                        image_array,
                        verbose=0
                    )

                    predicted_index = int(
                        np.argmax(predictions[0])
                    )

                    category = class_names[predicted_index]

                    confidence = float(
                        predictions[0][predicted_index]
                    ) * 100

                    # Display result
                    st.success("Classification Completed!")

                    st.markdown(
                        f"### Waste Category: {category.capitalize()}"
                    )

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.markdown("### Disposal Recommendation")

                    st.info(
                        advice.get(
                            category,
                            "Follow local waste disposal rules."
                        )
                    )

                    # Display all confidence scores
                    st.subheader("Confidence for Each Category")

                    for i, name in enumerate(class_names):

                        score = float(
                            predictions[0][i]
                        ) * 100

                        st.write(
                            f"{name.capitalize()}: {score:.2f}%"
                        )

                        st.progress(score / 100)

                    # Store history
                    st.session_state.history.append({
                        "Image": uploaded_file.name,
                        "Category": category.capitalize(),
                        "Confidence": f"{confidence:.2f}%",
                        "Date": datetime.now().strftime(
                            "%d-%m-%Y %H:%M"
                        )
                    })

                    st.caption(
                        "Predictions are estimates. "
                        "Please verify before disposing of waste."
                    )

# --------------------------------
# HISTORY PAGE
# --------------------------------

elif page == "Classification History":

    st.header("Classification History")

    if len(st.session_state.history) == 0:

        st.info("No classification history available yet.")

    else:

        for item in reversed(st.session_state.history):

            st.markdown(
                f"""
                **Image:** {item['Image']}

                **Category:** {item['Category']}

                **Confidence:** {item['Confidence']}

                **Date:** {item['Date']}
                """
            )

            st.divider()

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

# --------------------------------
# ABOUT PROJECT PAGE
# --------------------------------

elif page == "About Project":

    st.header("About the Project")

    st.write("""
    The Smart Waste Classification System uses
    Artificial Intelligence and Deep Learning to
    identify different categories of waste images.

    The system uses a MobileNetV2 transfer-learning
    model trained with a waste image dataset.
    """)

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - TensorFlow and Keras
    - MobileNetV2
    - Streamlit
    - NumPy
    - Pillow
    """)

    st.subheader("Waste Categories")

    for category in class_names:
        st.write("♻️ " + category.capitalize())

    st.subheader("Model Performance")

    st.write("Validation accuracy from training: 82.57%")

# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "Smart Waste Classification System | "
    "Python + TensorFlow + MobileNetV2 + Streamlit"
)
