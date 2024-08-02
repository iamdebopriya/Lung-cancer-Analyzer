import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the trained model
model = tf.keras.models.load_model('vgg16_model.zip')

# Set custom CSS for the Streamlit app
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #e6e6fa, #f4f4f9); /* Light purple to light gray gradient */
    }
    .stTitle {
        color: #4b0082; /* Title color */
        font-size: 2.5em; /* Title size */
        font-family: 'Arial', sans-serif;
    }
    .stButton > button {
        background-color: #4b0082; /* Button background color */
        color: #fff; /* Button text color */
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 1em;
    }
    .stButton > button:hover {
        background-color: #800080; /* Button hover color */
    }
    .stSidebar .sidebar-content {
        font-family: 'Arial', sans-serif;
    }
    .stMarkdown h2 {
        color: #4b0082; /* Heading color */
        font-family: 'Arial', sans-serif;
    }
    .stMarkdown h3 {
        color: #4b0082; /* Subheading color */
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Set the title
st.title("Chest Cancer Classification and Advice")

# Image uploader
uploaded_file = st.file_uploader("Choose a CT-Scan image...", type=["jpg", "png", "jpeg"])

# Sidebar for advice sections
st.sidebar.title("Advice Sections")
advice_option = st.sidebar.radio(
    "Select an Advice Category:",
    ["Adenocarcinoma", "Large Cell Carcinoma", "Squamous Cell Carcinoma", "Normal"]
)

# Define descriptions for cancer types
cancer_descriptions = {
    'Adenocarcinoma': {
        'description': "Adenocarcinoma is a type of cancer that forms in mucus-secreting glands and is commonly found in the lungs, breast, prostate, and other organs.",
        'location': "It can occur in various organs including the lungs, breast, prostate, and pancreas.",
        'causes': "Risk factors include smoking, family history of cancer, exposure to certain chemicals, and chronic inflammation."
    },
    'Large Cell Carcinoma': {
        'description': "Large cell carcinoma is a type of lung cancer characterized by large, abnormal cells. It is one of the subtypes of non-small cell lung cancer.",
        'location': "Typically found in the lungs.",
        'causes': "Major risk factors include smoking, exposure to radon gas, and a history of lung disease."
    },
    'Squamous Cell Carcinoma': {
        'description': "Squamous cell carcinoma is a cancer that originates in squamous cells, which are found in the skin and other tissues. It is commonly seen in the skin, lungs, and mouth.",
        'location': "Commonly occurs in the skin, lungs, and mouth.",
        'causes': "Risk factors include excessive sun exposure, smoking, and certain viral infections (e.g., HPV)."
    },
    'Normal': {
        'description': "The scanned image shows no signs of cancer. This result indicates a normal scan.",
        'location': "Not applicable.",
        'causes': "No cancer-related causes."
    }
}

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((150, 150))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict the class
    prediction = model.predict(image_array)
    classes = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Squamous Cell Carcinoma', 'Normal']
    predicted_class = classes[np.argmax(prediction)]

    # Debugging output to check the predicted_class
    st.write(f"Debug: Predicted class is '{predicted_class}'")

    # Display the prediction
    st.write(f"Predicted Class: **{predicted_class}**")

    # Display detailed description for the predicted cancer type
    if predicted_class in cancer_descriptions:
        st.write(f"### Details for {predicted_class}")
        st.write(f"**Description:** {cancer_descriptions[predicted_class]['description']}")
        st.write(f"**Location:** {cancer_descriptions[predicted_class]['location']}")
        st.write(f"**Causes:** {cancer_descriptions[predicted_class]['causes']}")
    else:
        st.write("Details for the predicted class are not available.")

    # Display advice based on the selected category from the sidebar
    if advice_option == 'Adenocarcinoma':
        st.sidebar.write("### Advice for Adenocarcinoma")
        st.sidebar.write("""
            - **Medications:** Follow the prescribed treatment plan, which may include chemotherapy, targeted therapy, or immunotherapy.
            - **Diet:** Maintain a balanced diet with plenty of fruits, vegetables, and lean proteins. Avoid smoking and alcohol.
            - **Exercise:** Engage in regular physical activity, such as walking or yoga, to maintain overall health and well-being.
            - **Support:** Join support groups or counseling sessions to cope with emotional stress and anxiety.
        """)
    elif advice_option == 'Large Cell Carcinoma':
        st.sidebar.write("### Advice for Large Cell Carcinoma")
        st.sidebar.write("""
            - **Medications:** Adhere to the treatment plan, which may involve chemotherapy, radiation therapy, or targeted therapy.
            - **Diet:** Consume a nutritious diet rich in antioxidants, vitamins, and minerals. Avoid processed foods and sugary drinks.
            - **Exercise:** Participate in moderate exercise routines to enhance physical strength and immunity.
            - **Support:** Seek psychological support to manage the emotional and mental aspects of the diagnosis.
        """)
    elif advice_option == 'Squamous Cell Carcinoma':
        st.sidebar.write("### Advice for Squamous Cell Carcinoma")
        st.sidebar.write("""
            - **Medications:** Follow the treatment regimen, which might include surgery, radiation therapy, or chemotherapy.
            - **Diet:** Eat a healthy diet with a focus on anti-inflammatory foods, such as berries, nuts, and green leafy vegetables.
            - **Exercise:** Stay active with regular exercise to boost your immune system and improve overall health.
            - **Support:** Utilize support networks, including family, friends, and cancer support groups, for emotional and practical assistance.
        """)
    else:
        st.sidebar.write("### Advice for Normal Diagnosis")
        st.sidebar.write("""
            - **General Health:** Continue with routine health check-ups and screenings as recommended by your healthcare provider.
            - **Diet:** Maintain a balanced diet with a variety of nutrients to support overall health.
            - **Exercise:** Engage in regular physical activity to keep your body fit and healthy.
            - **Well-being:** Focus on mental and emotional well-being through mindfulness practices and relaxation techniques.
        """)

    # Plotting analysis
    st.write("### Prediction Analysis")

    # Bar plot
    st.write("#### Bar Plot of Prediction Probabilities")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(classes, prediction[0], color='#4b0082')
    ax.set_xlabel('Cancer Type')
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities for Each Cancer Type')
    st.pyplot(fig)

    # Pie plot
    st.write("#### Pie Chart of Prediction Probabilities")
    fig, ax = plt.subplots(figsize=(12, 6))
    wedges, texts, autotexts = ax.pie(
        prediction[0],
        labels=classes,
        autopct='%1.1f%%',
        colors=['#4b0082', '#6a0dad', '#8a2be2', '#d8bfd8'],
        textprops=dict(color='black', fontsize=12)
    )
    ax.set_title('Distribution of Prediction Probabilities')

    # Adjusting label positions
    for text in texts:
        text.set_fontsize(12)
        text.set_color('black')
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_color('black')

    st.pyplot(fig)

    # 3D Plot
    st.write("#### 3D Bar Plot of Prediction Probabilities")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(len(classes))
    y = np.ones_like(x)
    z = np.zeros_like(x)
    dx = np.ones_like(x) * 0.5
    dy = np.ones_like(x) * 0.5
    dz = prediction[0]
    ax.bar3d(x, y, z, dx, dy, dz, color='#4b0082')
    ax.set_xlabel('Cancer Type')
    ax.set_ylabel('Value')
    ax.set_zlabel('Probability')
    ax.set_title('3D Bar Plot of Prediction Probabilities')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    st.pyplot(fig)
