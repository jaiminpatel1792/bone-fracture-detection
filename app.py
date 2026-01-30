import streamlit as st
from PIL import Image

from inference import load_model, predict_fracture, DEFAULT_THRESHOLD

st.set_page_config(page_title="Bone Fracture Detection", layout="centered")

st.title("Bone Fracture Detection (X-ray)")
st.write("Upload an X-ray image and the model will predict whether it is fractured or not.")

# Cache the model so it loads only once
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

threshold = st.slider("Decision threshold (fractured if probability ≥ threshold)",
                      0.05, 0.95, float(DEFAULT_THRESHOLD), 0.05)

uploaded = st.file_uploader("Upload X-ray image (jpg / png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded)

    st.image(img, caption="Uploaded X-ray", use_container_width=True)

    with st.spinner("Predicting..."):
        prob, label, name = predict_fracture(model, img, threshold=threshold)

    st.subheader("Prediction Result")
    st.metric("P(fractured)", f"{prob:.3f}")

    if label == 1:
        st.error(f"Prediction: **{name.upper()}**")
    else:
        st.success(f"Prediction: **{name.upper()}**")

st.markdown("---")
st.caption("Demo project. Model: DenseNet transfer learning.")
