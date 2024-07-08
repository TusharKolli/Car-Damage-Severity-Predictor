import streamlit as st
import numpy as np
import keras
import numpy as np
import keras
import keras.utils as im
import matplotlib.pyplot as plt
from keras.models import Model
from keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import pickle




def predict(model, image):
    # Preprocess the image
    image = image.convert('RGB')  # Convert to RGB to ensure three channels
    image = image.resize((224, 224))
    x = im.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)

    # Make predictions using the loaded model
    output = np.argmax(model.predict(img_data), axis=1)

    # Mapping the output index to class labels
    index = ['Minor', 'Moderate', 'Severe', 'Good']
    result = index[output[0]]

    return result



st.title("")

def main():
    st.title("Car Damage Severity Assessmnent System")

    # Add navigation to switch between pages
    page = st.sidebar.selectbox("Select a page", ["Model Prediction", "About the Model"])

    if page == "Model Prediction":
        model_prediction_page()
    elif page == "About the Model":
        about_model_page()

def model_prediction_page():
    st.header("Model Prediction")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image


        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Load the deep learning model
        
        model = load_model("C:/Users/Tushar/Desktop/MObile Net V2.model/MObile Net V2.model/MObile Net V2.model/MobileNet_Model_Final.keras")
   
        with st.spinner("Making prediction..."):
            prediction = predict(model, image)

        st.subheader("Prediction Results")
        st.write(prediction)

def about_model_page():
    st.header("About the Model")

    # Use st.columns to organize information into three columns
    col1, col2, col3 = st.columns(3)

    # Column 1: Information about the model
    col1.write("I employed a Transfer Learning approach using the DenseNet-201 model for this project.")
    col1.markdown("<br>", unsafe_allow_html=True)
    col1.subheader("Model Overview:")
    col1.write("The DenseNet-201 model is a deep neural network architecture known for its dense connections between layers.")
    col1.write("These connections facilitate feature reuse and enhance the model's ability to capture complex patterns in the data.")
    col1.write("The model is pre-trained on a large dataset, and I fine-tuned it for my specific task.")

    # Column 2: DenseNet-201 Model details
    col2.subheader("DenseNet-201 Architecture:")
    col2.write("DenseNet-201 consists of multiple densely connected blocks, each containing convolutional and pooling layers.")
    col2.write("The densely connected structure promotes gradient flow, mitigating the vanishing gradient problem.")
    col2.write("This architecture is particularly effective for image classification tasks, providing high accuracy.")

    # Column 3: Transfer Learning details
    col3.subheader("Transfer Learning Approach:")
    col3.write("Transfer learning involves leveraging the knowledge gained by a model on a large dataset for a specific task.")
    col3.write("For this project, I used the DenseNet-201 model pre-trained on ImageNet.")
    col3.write("I fine-tuned the model on my target dataset, allowing it to adapt to the specific features of my project.")

    # Add spacing between columns
    col1.markdown("<br>", unsafe_allow_html=True)
    col2.markdown("<br>", unsafe_allow_html=True)
    col3.markdown("<br>", unsafe_allow_html=True)

    # Additional details in Column 1
    col1.subheader("Model Training Details:")
    col1.write("During the fine-tuning process, I trained the model for a specified number of epochs.")
    col1.write("The learning rate, optimizer, and other hyperparameters were tuned to achieve optimal performance.")
    col1.write("Data augmentation techniques were applied to enhance the model's ability to generalize.")

    # Additional details in Column 2
    col2.subheader("Performance Metrics:")
    col2.write("To evaluate the model's performance, I used metrics such as accuracy, precision, recall, and F1 score.")
    col2.write("Validation and test datasets were employed to assess the model's generalization ability.")
    col2.write("The model's performance was analyzed in terms of both quantitative metrics and qualitative assessments.")

    # Additional details in Column 3
    col3.subheader("Future Improvements:")
    col3.write("In future iterations, I plan to explore additional fine-tuning strategies.")
    col3.write("Hyperparameter tuning, model architecture modifications, and the incorporation of ensemble methods are areas of potential improvement.")
    col3.write("Continued monitoring and updating of the model will be carried out to adapt to evolving data patterns.")

if __name__ == "__main__":
    main()