import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array

# Define CNN models for knee osteoarthritis and lung cancer detection
def create_knee_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(250, 250, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(5, activation='softmax')  # Assuming 5 classes for knee severity
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_lung_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(4, activation='softmax')  # 4 classes for lung cancer
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

knee_model = create_knee_model()
lung_model = create_lung_model()

# Define class labels and advice
knee_classes = ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe']
lung_classes = ['Adenocarcinoma', 'Squamous Cell Carcinoma', 'Large Cell Carcinoma', 'Normal']

knee_advice = {
    'Normal': {
        'Doctors': "Routine check-ups are sufficient.",
        'Medication': "No medication needed.",
        'Surgery': "No surgery needed.",
        'Diets': "Maintain a balanced diet for overall health."
    },
    'Doubtful': {
        'Doctors': "Consult an orthopedic specialist.",
        'Medication': "Consider anti-inflammatory medications.",
        'Surgery': "Surgery is generally not recommended at this stage.",
        'Diets': "Maintain a healthy weight and consider anti-inflammatory diets."
    },
    'Mild': {
        'Doctors': "Regular check-ups with a specialist.",
        'Medication': "Mild pain relievers and joint health supplements.",
        'Surgery': "Surgery is usually not needed at this stage.",
        'Diets': "Adopt a balanced diet with focus on joint health."
    },
    'Moderate': {
        'Doctors': "Consult a specialist for a detailed treatment plan.",
        'Medication': "Medication may include pain relievers and joint supplements.",
        'Surgery': "Surgery options may be discussed based on progression.",
        'Diets': "Focus on a diet rich in anti-inflammatory foods and joint health supplements."
    },
    'Severe': {
        'Doctors': "Immediate consultation with a specialist is advised.",
        'Medication': "Strong medications and possibly injections for pain relief.",
        'Surgery': "Surgical options such as knee replacement may be considered.",
        'Diets': "Follow a strict diet plan to manage inflammation and support joint health."
    }
}

lung_advice = {
    'Adenocarcinoma': {
        'Doctors': "Seek consultation with an oncologist specializing in lung cancer.",
        'Medication': "Treatment may involve targeted therapies and chemotherapy.",
        'Surgery': "Surgical options may be considered based on the stage.",
        'Diets': "Follow a diet rich in antioxidants and consult a nutritionist."
    },
    'Squamous Cell Carcinoma': {
        'Doctors': "Consultation with a specialized oncologist is advised.",
        'Medication': "Chemotherapy and targeted therapies may be recommended.",
        'Surgery': "Surgery may be necessary depending on the stage.",
        'Diets': "Focus on a diet that supports overall health and aids recovery."
    },
    'Large Cell Carcinoma': {
        'Doctors': "Consult with an oncologist for a comprehensive treatment plan.",
        'Medication': "Chemotherapy and targeted treatments may be prescribed.",
        'Surgery': "Surgical interventions may be considered based on progression.",
        'Diets': "A balanced diet to support health and recovery is recommended."
    },
    'Normal': {
        'Doctors': "Regular check-ups are advised.",
        'Medication': "No medication needed.",
        'Surgery': "No surgery needed.",
        'Diets': "Maintain a balanced diet and avoid smoking."
    }
}

# Set up Streamlit page
st.set_page_config(page_title="Medical Diagnostic App", page_icon="🩺")

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #f8a5c2, #ffffff); /* Gradient from light pink to purple */
    }
    .stTitle {
        color: #4b0082; /* Title color */
        font-size: 2.5em; /* Title size */
        font-family: 'Arial', sans-serif;
    }
    .stSidebar {
        background-color: #6c5ce7; /* Purple background for the sidebar */
    }
    .stSidebar .sidebar-content {
        color: #ffffff; /* White text color in the sidebar */
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

# Sidebar for selection
st.sidebar.title("Select Task")
task = st.sidebar.selectbox("Choose a diagnostic task", ["View Advice", "Knee Osteoarthritis Severity Prediction", "Lung Cancer Detection"])

if task == "Knee Osteoarthritis Severity Prediction":
    st.title("Knee Osteoarthritis Severity Prediction")

    # Image uploader for knee X-ray
    uploaded_knee_file = st.file_uploader("Choose a knee X-ray image...", type=["jpg", "png", "jpeg"], key='knee')

    if uploaded_knee_file is not None:
        # Preprocess the image
        image = Image.open(uploaded_knee_file).convert('L')  # Convert to grayscale
        image = image.resize((250, 250))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict the severity
        severity_prediction = knee_model.predict(image_array)
        severity_class = knee_classes[np.argmax(severity_prediction)]

        # Display the uploaded image
        st.image(uploaded_knee_file, caption='Uploaded Knee X-ray Image', use_column_width=True)
        st.write(f"**Predicted Severity:** {severity_class}")

        # Provide advice based on severity
        if severity_class in knee_advice:
            st.write(f"### Advice for {severity_class} Severity")
            st.write(f"**Doctors:** {knee_advice[severity_class]['Doctors']}")
            st.write(f"**Medication:** {knee_advice[severity_class]['Medication']}")
            st.write(f"**Surgery:** {knee_advice[severity_class]['Surgery']}")
            st.write(f"**Diets:** {knee_advice[severity_class]['Diets']}")

        # Plotting analysis
        st.write("### Prediction Analysis")

        # Bar plot
        st.write("#### Bar Plot of Prediction Probabilities")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(knee_classes, severity_prediction[0], color='#4b0082')
        ax.set_xlabel('Severity Class')
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities for Knee Osteoarthritis Severity')
        st.pyplot(fig)

        # Pie chart
        st.write("#### Pie Chart of Prediction Probabilities")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(severity_prediction[0], labels=knee_classes, autopct='%1.1f%%', colors=['#f8a5c2', '#ff6b81', '#ff4757', '#ff6348', '#ff7f50'])
        ax.set_title('Prediction Probabilities for Knee Osteoarthritis Severity')
        st.pyplot(fig)

if task == "Lung Cancer Detection":
    st.title("Lung Cancer Detection")

    # Image uploader for lung X-ray
    uploaded_lung_file = st.file_uploader("Choose a lung X-ray image...", type=["jpg", "png", "jpeg"], key='lung')

    if uploaded_lung_file is not None:
        # Preprocess the image
        image = Image.open(uploaded_lung_file).convert('RGB')  # Convert to RGB
        image = image.resize((150, 150))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict the type of lung cancer
        cancer_prediction = lung_model.predict(image_array)
        cancer_class = lung_classes[np.argmax(cancer_prediction)]

        # Display the uploaded image
        st.image(uploaded_lung_file, caption='Uploaded Lung X-ray Image', use_column_width=True)
        st.write(f"**Predicted Class:** {cancer_class}")

        # Provide advice based on lung cancer type
        if cancer_class in lung_advice:
            st.write(f"### Advice for {cancer_class}")
            st.write(f"**Doctors:** {lung_advice[cancer_class]['Doctors']}")
            st.write(f"**Medication:** {lung_advice[cancer_class]['Medication']}")
            st.write(f"**Surgery:** {lung_advice[cancer_class]['Surgery']}")
            st.write(f"**Diets:** {lung_advice[cancer_class]['Diets']}")

        # Plotting analysis
        st.write("### Prediction Analysis")

        # Bar plot
        st.write("#### Bar Plot of Prediction Probabilities")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(lung_classes, cancer_prediction[0], color='#4b0082')
        ax.set_xlabel('Cancer Class')
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities for Lung Cancer Detection')
        st.pyplot(fig)

        # Pie chart
        st.write("#### Pie Chart of Prediction Probabilities")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(cancer_prediction[0], labels=lung_classes, autopct='%1.1f%%', colors=['#f8a5c2', '#ff6b81', '#ff4757', '#ff6348'])
        ax.set_title('Prediction Probabilities for Lung Cancer Detection')
        st.pyplot(fig)

if task == "View Advice":
    st.title("View Advice")
    st.write("### Choose a condition to view advice")

    condition = st.selectbox("Select a condition", ["Knee Osteoarthritis", "Lung Cancer"])

    if condition == "Knee Osteoarthritis":
        severity = st.selectbox("Select severity level", knee_classes)
        if severity:
            st.write(f"### Advice for {severity} Severity")
            st.write(f"**Doctors:** {knee_advice[severity]['Doctors']}")
            st.write(f"**Medication:** {knee_advice[severity]['Medication']}")
            st.write(f"**Surgery:** {knee_advice[severity]['Surgery']}")
            st.write(f"**Diets:** {knee_advice[severity]['Diets']}")

    if condition == "Lung Cancer":
        cancer_type = st.selectbox("Select cancer type", lung_classes)
        if cancer_type:
            st.write(f"### Advice for {cancer_type}")
            st.write(f"**Doctors:** {lung_advice[cancer_type]['Doctors']}")
            st.write(f"**Medication:** {lung_advice[cancer_type]['Medication']}")
            st.write(f"**Surgery:** {lung_advice[cancer_type]['Surgery']}")
            st.write(f"**Diets:** {lung_advice[cancer_type]['Diets']}")


st.write("Made with ❤️ by HealthAI")
