# --- Conditional Package Installation ---
# This block attempts to import necessary libraries. If any are missing,
# it will try to install them via pip. This is useful for self-contained
# execution environments like new Colab sessions.
# For production deployments (e.g., Streamlit Cloud), it's generally
# recommended to rely solely on requirements.txt for dependency management.
try:
    import streamlit as st
    import numpy as np
    from PIL import Image
    import io
    import os
    import time
    import tensorflow as tf
    from dotenv import load_dotenv
    from pyngrok import ngrok # Required for Colab/remote public URL
    import subprocess
    import threading
except ImportError:
    import subprocess
    import sys
    st.warning("Some required packages are not found. Attempting to install them...")
    required_packages = [
        "streamlit",
        "numpy",
        "Pillow",
        "tensorflow",
        "python-dotenv",
        "pyngrok"
    ]
    try:
        for package in required_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        st.success("All required packages installed! Please refresh the page or rerun the app.")
        # Streamlit needs to re-run after installs. Using st.stop() or a message is good.
        st.stop()
    except Exception as e:
        st.error(f"Failed to install packages: {e}. Please install them manually using `pip install -r requirements.txt`.")
        st.stop()

# --- Load environment variables from .env file ---
load_dotenv()
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")

# --- PharyngitisModel Class (Integrate Your Actual Model Here) ---
class PharyngitisModel:
    def __init__(self):
        self.model = None
        st.session_state.get('model_status', 'Model not initialized.')

    def load_model(self, model_path):
        st.session_state.model_status = f"Attempting to load model from: '{model_path}'"
        try:
            # --- YOUR ACTUAL MODEL LOADING CODE GOES HERE ---
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                st.session_state.model_status = "Model loaded successfully."
            else:
                if "best_pharyngitis_model_fold_" in model_path:
                    self.model = True # Simulate loaded model
                    st.session_state.model_status = "Model loaded successfully (simulated - real file not found)."
                else:
                    st.session_state.model_status = f"Error: Model file '{model_path}' not found."
                    self.model = None
        except Exception as e:
            st.session_state.model_status = f"Error loading model: {e}"
            self.model = None

    def predict_with_features(self, image_array):
        """
        Performs prediction on the given preprocessed image array.
        --- REPLACE THIS SECTION WITH YOUR ACTUAL MODEL PREDICTION LOGIC ---
        """
        if self.model is None or self.model is True:
            if self.model is True: # Simulated model
                st.session_state.prediction_status = "Simulating prediction (real model not loaded)."
                time.sleep(2)
                prediction_probability = np.random.rand(1, 1)[0][0]
                confidence_score = 0.8 + (np.random.rand() * 0.2)
                features = np.random.rand(1, 256)
                return prediction_probability, confidence_score, features
            else:
                raise ValueError("Model not loaded. Please call load_model() first.")

        st.session_state.prediction_status = f"Received image array for prediction with shape: {image_array.shape}"

        # --- YOUR ACTUAL MODEL PREDICTION CODE GOES HERE ---
        try:
            predictions = self.model.predict(image_array)
            prediction_probability = predictions[0][1] if predictions.shape[1] > 1 else predictions[0][0]
            confidence_score = prediction_probability
            features = np.random.rand(1, 256) # Replace with actual feature extraction if applicable

            st.session_state.prediction_status = "Prediction generated successfully."
            return prediction_probability, confidence_score, features
        except Exception as e:
            st.session_state.prediction_status = f"Error during actual prediction: {e}"
            raise e


# --- Image Relevance Check (Placeholder) ---
def is_image_relevant(image: Image.Image) -> bool:
    """
    Placeholder function to check if the image is 'relevant' (e.g., a throat image).
    For demonstration, this function always returns True.
    """
    return True


# --- Streamlit Application Layout ---

st.set_page_config(
    page_title="Pharyngitis Detector",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("⚕️ Pharyngitis Detector App")
st.markdown("""
Upload an image of a throat, or take a real-time photo. The AI model will provide a
simulated or actual prediction regarding the likelihood of pharyngitis.
""")

# --- AI Disclaimer ---
st.warning("""
**Disclaimer:** This application uses an AI model for **demonstration and informational purposes only**.
It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the
advice of a qualified healthcare provider for any medical concerns. Do not disregard professional
medical advice or delay in seeking it because of information presented by this AI.
""")

if 'model_instance' not in st.session_state:
    st.session_state.model_instance = PharyngitisModel()
    st.session_state.model_instance.load_model('best_pharyngitis_model_fold_0.h5')

st.info(st.session_state.get('model_status', 'Initializing model...'))

# --- Image Input Option ---
input_method = st.radio(
    "Choose input method:",
    ("Upload Image", "Take Photo"),
    horizontal=True
)

uploaded_file = None
camera_photo = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
else: # "Take Photo"
    camera_photo = st.camera_input("Take a photo")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
elif camera_photo is not None:
    image = Image.open(camera_photo)
    st.image(image, caption='Taken Photo.', use_column_width=True)

prediction_placeholder = st.empty()

if image is not None:
    st.write("") # Add a little space

    if st.button("Get Prediction"):
        if st.session_state.model_instance.model is None and st.session_state.model_instance.model is not True:
            st.error("Model not loaded. Please check the model path and ensure it's available.")
        else:
            # --- Check Image Relevance ---
            if not is_image_relevant(image):
                st.error("The submitted image does not appear to be a relevant throat image. Please upload or take a photo of a throat.")
            else:
                with st.spinner('Processing image and getting prediction...'):
                    try:
                        img_array = np.array(image)

                        if img_array.ndim == 2:
                            img_array = np.stack((img_array,)*3, axis=-1)
                        elif img_array.shape[2] == 4:
                            img_array = img_array[:, :, :3]

                        img_resized = Image.fromarray(img_array).resize((224, 224))
                        img_processed = np.array(img_resized).astype(np.float32)
                        img_processed /= 255.0

                        img_for_prediction = np.expand_dims(img_processed, axis=0)

                        prediction_score, confidence_score, extracted_features = st.session_state.model_instance.predict_with_features(img_for_prediction)

                        if prediction_score > 0.7:
                            interpretation = "High probability of Pharyngitis detected."
                            st.error(f"**Prediction:** {interpretation}")
                        elif prediction_score > 0.4:
                            interpretation = "Moderate probability of Pharyngitis detected."
                            st.warning(f"**Prediction:** {interpretation}")
                        else:
                            interpretation = "Low probability of Pharyngitis detected."
                            st.success(f"**Prediction:** {interpretation}")

                        st.markdown(f"**Confidence Score:** {confidence_score:.2f}")
                        st.markdown(f"*(Simulated features shape: {extracted_features.shape})*")

                        st.subheader("About Pharyngitis (Sore Throat)")
                        st.markdown("""
                        Pharyngitis, commonly known as a sore throat, is an inflammation of the pharynx (the back of the throat).
                        It often causes discomfort, scratchiness, or pain, especially when swallowing. It's a very common
                        condition and usually not a cause for serious concern.
                        """)

                        st.markdown("---")
                        st.subheader("Common Causes:")
                        st.markdown("""
                        * **Viral Infections:** Most sore throats are caused by viruses, such as those that cause the common cold, flu, or mononucleosis.
                        * **Bacterial Infections:** Less commonly, it can be caused by bacteria, with Group A Streptococcus (strep throat) being a well-known example. Bacterial pharyngitis may require antibiotics.
                        * **Allergies:** Post-nasal drip from allergies can irritate the throat.
                        * **Dry Air:** Breathing dry air, especially indoors during winter or with mouth breathing, can lead to dryness and irritation.
                        * **Irritants:** Exposure to smoke, pollution, or chemical irritants.
                        * **Voice Strain:** Overuse or misuse of the voice.
                        """)

                        st.markdown("---")
                        st.subheader("General Tips & Home Remedies (from common knowledge):")
                        st.markdown("""
                        * **Rest:** Get plenty of rest, especially for your voice.
                        * **Stay Hydrated:** Drink plenty of fluids like water, warm tea with honey, or clear broths. Avoid caffeine and alcohol.
                        * **Gargle with Salt Water:** Mix 1/4 to 1/2 teaspoon of salt in 1 cup (250 mL) of warm water and gargle. This can help soothe the throat and reduce inflammation.
                        * **Honey:** Honey can help soothe the throat. Mix it with warm water or tea. (Note: Do not give honey to infants under 1 year old due to the risk of botulism).
                        * **Lozenges or Hard Candy:** These can help stimulate saliva production, which keeps the throat moist and can relieve pain.
                        * **Humidifier:** Use a cool-mist humidifier to add moisture to the air, which can alleviate dryness and irritation.
                        * **Avoid Irritants:** Steer clear of smoke, very spicy foods, and very hot liquids.
                        * **Over-the-Counter Pain Relievers:** Medications like acetaminophen (paracetamol) or ibuprofen can help relieve pain and fever.
                        """)

                        st.markdown("---")
                        st.subheader("When to See a Doctor:")
                        st.markdown("""
                        It's advisable to consult a healthcare professional if you experience:
                        * A sore throat lasting longer than a week or getting worse.
                        * Severe difficulty swallowing or breathing.
                        * High fever (above 100.4°F or 38°C).
                        * Swollen, tender glands in your neck.
                        * A rash or white patches on your tonsils.
                        * Blood in your saliva or phlegm.
                        """)


                    except ValueError as ve:
                        st.error(f"Prediction Error: {ve}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
else:
    prediction_placeholder.empty()

# --- Code below is for running Streamlit in Google Colab (if applicable) ---
# If you're running locally and don't need a public URL, you can remove this section.
if NGROK_AUTH_TOKEN:
    # Ensure ngrok is imported if not already during the initial try block
    try:
        from pyngrok import ngrok
        import subprocess
        import threading
    except ImportError:
        st.error("pyngrok is required for public URL tunneling but could not be imported.")
        st.stop() # Stop if ngrok isn't available for Colab execution.

    # Start Streamlit in a separate thread
    def run_streamlit():
        # Use --server.enableCORS false and --server.enableXsrfProtection false for Colab compatibility
        subprocess.run(["streamlit", "run", "app.py", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"])

    # Only start the thread and ngrok if not already running (to prevent multiple tunnels)
    if 'streamlit_thread_started' not in st.session_state:
        streamlit_thread = threading.Thread(target=run_streamlit)
        streamlit_thread.start()
        st.session_state.streamlit_thread_started = True # Mark thread as started

        try:
            ngrok.set_auth_token(NGROK_AUTH_TOKEN)
            public_url = ngrok.connect(8501) # Streamlit's default port is 8501
            st.success(f"Your Streamlit app is live at: {public_url}")
            print(f"Your Streamlit app is live at: {public_url}") # For Colab output
        except Exception as e:
            st.error(f"Error creating ngrok tunnel: {e}")
            print(f"Error creating ngrok tunnel: {e}")
else:
    st.warning("NGROK_AUTH_TOKEN not found in .env. Public URL tunneling via ngrok will not be available.")
    print("NGROK_AUTH_TOKEN not found in .env. Cannot create public URL.")
    print("Please add NGROK_AUTH_TOKEN='YOUR_TOKEN_HERE' to your .env file for public access in Colab.")
