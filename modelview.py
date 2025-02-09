import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")
#import warnings
#import streamlit as st
#st.set_option('deprecation.showPyplotGlobalUse', False)
#import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

warnings.simplefilter(action='ignore', category=DeprecationWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Load the denoising model
#@st.cache(allow_output_mutation=True)
@st.cache_resource

def load_denoising_model():
    return load_model('image_denoisingy.keras', compile=False)

model = load_denoising_model()

# Preprocess the input image
def preprocess_image(uploaded_image):
    try:
        img = Image.open(uploaded_image).convert('L')  # Convert to grayscale
        img_resized = img.resize((128, 128))  # Resize to model input size
        img_array = np.array(img_resized) / 255.0  # Normalize pixel values
        img_array = img_array.reshape(1, 128, 128, 1).astype('float32')  # Add batch and channel dimensions
        return img, img_array
    except Exception as e:
        st.error(f"Invalid file format: {e}")
        st.stop()

# Postprocess and save the denoised image
def postprocess_and_save_denoised_image(denoised_output):
    denoised_image = denoised_output.reshape(128, 128)  # Remove batch and channel dimensions
    denoised_image = (denoised_image * 255).astype(np.uint8)  # Convert back to 0-255 range
    return Image.fromarray(denoised_image)

# Get image bytes
def get_image_bytes(image):
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    return img_bytes.getvalue()

# Streamlit UI
st.title("Image Denoising App")
st.write("Upload a noisy grayscale image, and this app will denoise it using the trained model.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Show the uploaded image
    st.subheader("Uploaded Image:")
    img, img_array = preprocess_image(uploaded_file)
    
    # Display original and denoised images side by side
    col1, col2 = st.columns(2)
    col1.image(img, caption="Noisy Image", use_container_width=True)
    
    # Predict (Denoise)
    with st.spinner('Denoising the image...'):
        denoised_output = model.predict(img_array)
        denoised_image = postprocess_and_save_denoised_image(denoised_output)
    
    col2.image(denoised_image, caption="Denoised Output", use_container_width=True)

    # Provide a download link for the denoised image
    st.download_button(
        label="Download Denoised Image",
        data=get_image_bytes(denoised_image),
        file_name="denoised_image.png",
        mime="image/png"
    )
