import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model , Model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import heapq


@st.cache_resource
def load_model():
    return pickle.load(open("Inception_v3.pkl", "rb"))

model = load_model()

# Function to preprocess the image
def preprocess_image(image_file, target_size=(299, 299)):
    img = load_img(image_file, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


# Function to normalize feature vectors
def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

# Function to calculate similarity scores
def find_top_k_similar(query_features, feature_database, k=5, method="cosine"):
    similarities = []
    for entry in feature_database:
        features = np.array(entry["features"])
        image_path = entry["image_path"]

        # Compute similarity
        if method == "cosine":
            similarity = cosine_similarity([query_features], [features])[0][0]
        elif method == "euclidean":
            similarity = 1 / (1 + np.linalg.norm(query_features - features))
        elif method == "manhattan":
            similarity = 1 / (1 + np.sum(np.abs(query_features - features)))
        else:
            raise ValueError("Invalid method. Choose 'cosine', 'euclidean', or 'manhattan'.")

        similarities.append((similarity, image_path))
    
    return heapq.nlargest(k, similarities, key=lambda x: x[0])

# Load feature database for a specific class
def load_feature_database(class_name):
    file_path = os.path.join("class_features", f"{class_name}_features.pkl")
    if not os.path.exists(file_path):
        st.error(f"Feature database for class '{class_name}' not found.")
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Streamlit app
st.title("👗 Dynamic Fashion Recommender System 🛒")
st.write(
    "Welcome to the fashion shop! Upload a fashion product, and we'll classify it and recommend similar items for you."
)

uploaded_file = st.file_uploader("Upload your fashion image (JPG or PNG)", type=["jpg", "png"])
similarity_method = st.selectbox("Select Similarity Method", ["cosine", "euclidean", "manhattan"])
top_k = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

if uploaded_file:
    # Save uploaded file temporarily
    temp_file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    query_image = preprocess_image(temp_file_path)

    # Predict the class
    predictions = model.predict(query_image)
    predicted_class_index = np.argmax(predictions)

    # Class mapping (ensure this matches your training)
    class_labels = ['boot', 'dress', 'hat', 'outerwear', 'pants', 'sandal', 'shoe', 'shorts', 'skirt', 'upperwear']
    predicted_class_label = class_labels[predicted_class_index]

    st.write(f"Predicted Class: **{predicted_class_label}**")

    feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
    # Load the feature database for the predicted class
    feature_database = load_feature_database(predicted_class_label)
    if feature_database is None:
        st.error("Could not load feature database for the predicted class.")
    else:
        # Extract features from the query image
        query_features = feature_extractor.predict(query_image).flatten()
        query_features = normalize_vector(query_features)

        # Find top-k similar items
        st.write(f"Finding top {top_k} similar items using {similarity_method.capitalize()} similarity...")
        top_similar_images = find_top_k_similar(query_features, feature_database, k=top_k, method=similarity_method)

        # Display recommendations
        st.write("### Recommended Items:")
        cols = st.columns(len(top_similar_images))
        for col, (score, image_path) in zip(cols, top_similar_images):
            with col:
                st.image(image_path, caption=f"Similarity: {score:.2f}", use_column_width=True)

    # Clean up temporary file
    os.remove(temp_file_path)
else:
    st.info("Please upload an image to get recommendations.")
