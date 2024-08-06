import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array




# Define CNN models for knee osteoarthritis, lung cancer, and chest disease detection
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

def create_chest_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(14, activation='softmax')  # 14 classes for chest diseases
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

knee_model = create_knee_model()
lung_model = load_model("vgg16_model.h5")
chest_model = create_chest_model()

# Define class labels and advice
knee_classes = ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe']
lung_classes = ['Adenocarcinoma', 'Squamous Cell Carcinoma', 'Large Cell Carcinoma', 'Normal']
chest_classes = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

knee_advice = {
    'Normal': {
        'Doctors': "Regular check-ups are recommended.",
        'Medication': "No medication needed.",
        'Surgery': "No surgery needed.",
        'Diets': "Maintain a healthy diet."
    },
    'Doubtful': {
        'Doctors': "Consult an orthopedic specialist.",
        'Medication': "Pain relievers if necessary.",
        'Surgery': "Not typically required at this stage.",
        'Diets': "Balanced diet with anti-inflammatory foods."
    },
    'Mild': {
        'Doctors': "Consult an orthopedic specialist.",
        'Medication': "Pain relievers and anti-inflammatory drugs.",
        'Surgery': "Rarely required.",
        'Diets': "Include omega-3 fatty acids in your diet."
    },
    'Moderate': {
        'Doctors': "Consult an orthopedic specialist.",
        'Medication': "Pain relievers, anti-inflammatory drugs, and possibly physical therapy.",
        'Surgery': "May be considered in some cases.",
        'Diets': "Balanced diet with plenty of vitamins and minerals."
    },
    'Severe': {
        'Doctors': "Consult an orthopedic specialist urgently.",
        'Medication': "Strong pain relievers, anti-inflammatory drugs.",
        'Surgery': "Likely required (e.g., knee replacement).",
        'Diets': "Nutritious diet to support recovery."
    }
}

lung_advice = {
    'Adenocarcinoma': {
        'Doctors': "Consult an oncologist.",
        'Medication': "Chemotherapy, targeted therapy.",
        'Surgery': "May require surgical intervention.",
        'Diets': "High-protein diet to maintain strength."
    },
    'Squamous Cell Carcinoma': {
        'Doctors': "Consult an oncologist.",
        'Medication': "Chemotherapy, targeted therapy.",
        'Surgery': "Surgery may be necessary.",
        'Diets': "Balanced diet, avoid smoking."
    },
    'Large Cell Carcinoma': {
        'Doctors': "Consult an oncologist.",
        'Medication': "Chemotherapy, immunotherapy.",
        'Surgery': "Surgery may be an option.",
        'Diets': "Healthy diet to support immune system."
    },
    'Normal': {
        'Doctors': "No need for medical consultation.",
        'Medication': "No medication needed.",
        'Surgery': "No surgery needed.",
        'Diets': "Maintain a healthy lifestyle."
    }
}

chest_advice = {
    'Atelectasis': {
        'Doctors': "Consult a pulmonologist.",
        'Medication': "Bronchodilators or antibiotics may be prescribed.",
        'Surgery': "Possible surgical intervention if severe.",
        'Diets': "Healthy diet, avoid smoking."
    },
    'Cardiomegaly': {
        'Doctors': "Consult a cardiologist.",
        'Medication': "Medication for underlying heart condition.",
        'Surgery': "Surgery may be required depending on the cause.",
        'Diets': "Low sodium diet, heart-healthy foods."
    },
    'Effusion': {
        'Doctors': "Consult a pulmonologist.",
        'Medication': "Diuretics or other medication to manage fluid levels.",
        'Surgery': "Thoracentesis or pleurodesis may be required.",
        'Diets': "Low sodium diet, stay hydrated."
    },
    'Infiltration': {
        'Doctors': "Consult a pulmonologist.",
        'Medication': "Antibiotics or antifungals depending on the cause.",
        'Surgery': "Rarely required.",
        'Diets': "Balanced diet with plenty of fluids."
    },
    'Mass': {
        'Doctors': "Consult an oncologist.",
        'Medication': "Depends on the nature of the mass (e.g., chemotherapy).",
        'Surgery': "May require surgical removal.",
        'Diets': "Nutrient-rich diet to support treatment."
    },
    'Nodule': {
        'Doctors': "Consult a pulmonologist or oncologist.",
        'Medication': "Depends on whether the nodule is benign or malignant.",
        'Surgery': "May require biopsy or removal.",
        'Diets': "Balanced diet, avoid smoking."
    },
    'Pneumonia': {
        'Doctors': "Consult a physician or pulmonologist.",
        'Medication': "Antibiotics, antivirals, or antifungals.",
        'Surgery': "Not typically required.",
        'Diets': "Stay hydrated, nutritious diet."
    },
    'Pneumothorax': {
        'Doctors': "Consult a pulmonologist or emergency care.",
        'Medication': "Pain relief medication.",
        'Surgery': "May require chest tube or surgery.",
        'Diets': "Healthy diet, avoid smoking."
    },
    'Consolidation': {
        'Doctors': "Consult a pulmonologist.",
        'Medication': "Antibiotics or antifungals.",
        'Surgery': "Rarely required.",
        'Diets': "Balanced diet with plenty of fluids."
    },
    'Edema': {
        'Doctors': "Consult a cardiologist or nephrologist.",
        'Medication': "Diuretics or other medications.",
        'Surgery': "Rarely required.",
        'Diets': "Low sodium diet, stay hydrated."
    },
    'Emphysema': {
        'Doctors': "Consult a pulmonologist.",
        'Medication': "Bronchodilators, inhaled steroids.",
        'Surgery': "Lung volume reduction surgery in severe cases.",
        'Diets': "High-calorie, nutrient-dense diet."
    },
    'Fibrosis': {
        'Doctors': "Consult a pulmonologist.",
        'Medication': "Antifibrotic agents, oxygen therapy.",
        'Surgery': "Lung transplant in advanced cases.",
        'Diets': "Balanced diet, avoid smoking."
    },
    'Pleural_Thickening': {
        'Doctors': "Consult a pulmonologist.",
        'Medication': "Depends on underlying cause.",
        'Surgery': "Possible in severe cases.",
        'Diets': "Healthy diet, avoid smoking."
    },
    'Hernia': {
        'Doctors': "Consult a surgeon.",
        'Medication': "Pain relief if necessary.",
        'Surgery': "May require surgical repair.",
        'Diets': "Avoid heavy meals, maintain a healthy weight."
    }
}

st.set_page_config(layout='wide')

# Define app layout and tasks
st.title("Medical Image Diagnosis and Advice System")

st.sidebar.title("Choose a Task")
task = st.sidebar.radio("", ("Knee Osteoarthritis Detection", "Lung Cancer Detection", "Chest Disease Detection"))
st.sidebar.title("About the App")
st.sidebar.write("""
### About the App

This application is designed to assist in diagnosing medical conditions from images. It currently supports three types of diagnostics:

1. **Knee Osteoarthritis Detection**: Analyze knee X-ray images to determine the severity of osteoarthritis, from normal to severe. The app provides recommendations for doctors, medications, and treatments based on the severity of the condition.

2. **Lung Cancer Detection**: Evaluate lung X-ray images to identify different types of lung cancer. It offers advice on treatments and follow-ups based on the type of cancer detected.

3. **Chest Disease Detection**: Detect various chest diseases using X-ray images. The app gives advice on treatment and lifestyle changes based on the detected disease.

**Features**:
- Image upload and preprocessing.
- Prediction and classification using pre-trained models.


- Detailed advice on treatment, medication, and lifestyle changes.

**How It Works**:
1. Upload an X-ray image.
2. The app preprocesses and classifies the image using machine learning models.
3. Receive personalized advice based on the diagnosis.
""")



if task == "Knee Osteoarthritis Detection":
    st.title("Knee Osteoarthritis Severity Detection")

    # Image uploader for knee X-ray
    uploaded_knee_file = st.file_uploader("Choose a knee X-ray image...", type=["jpg", "png", "jpeg"], key='knee')

    if uploaded_knee_file is not None:
        # Preprocess the image
        image = Image.open(uploaded_knee_file).convert('L')
        image = image.resize((250, 250))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict the knee osteoarthritis severity
        knee_prediction = knee_model.predict(image_array)
        knee_class = knee_classes[np.argmax(knee_prediction)]

        # Display the uploaded image
        st.image(uploaded_knee_file, caption='Uploaded Knee X-ray Image', use_column_width=True)
        st.write(f"**Predicted Severity:** {knee_class}")

        # Provide advice based on knee severity
        if knee_class in knee_advice:
            st.write(f"### Advice for {knee_class}")
            st.write(f"**Doctors:** {knee_advice[knee_class]['Doctors']}")
            st.write(f"**Medication:** {knee_advice[knee_class]['Medication']}")
            st.write(f"**Surgery:** {knee_advice[knee_class]['Surgery']}")
            st.write(f"**Diets:** {knee_advice[knee_class]['Diets']}")

        

if task == "Lung Cancer Detection":
    st.title("Lung Cancer Detection")

    # Image uploader for lung X-ray
    uploaded_lung_file = st.file_uploader("Choose a lung X-ray image...", type=["jpg", "png", "jpeg"], key='lung')
    if uploaded_lung_file is not None:
        # Preprocess the image
        image = Image.open(uploaded_lung_file).convert('RGB')  # Convert to RGB
        image = image.resize((150, 150))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Predict the lung cancer type
        lung_prediction = lung_model.predict(image_array)
        lung_class = lung_classes[np.argmax(lung_prediction)]

        # Display the uploaded image
        st.image(uploaded_lung_file, caption='Uploaded Lung X-ray Image', use_column_width=True)
        st.write(f"**Predicted Cancer Type:** {lung_class}")

        # Provide advice based on cancer type
        if lung_class in lung_advice:
            st.write(f"### Advice for {lung_class}")
            st.write(f"**Doctors:** {lung_advice[lung_class]['Doctors']}")
            st.write(f"**Medication:** {lung_advice[lung_class]['Medication']}")
            st.write(f"**Surgery:** {lung_advice[lung_class]['Surgery']}")
            st.write(f"**Diets:** {lung_advice[lung_class]['Diets']}")

        
        

if task == "Chest Disease Detection":
    st.title("Chest Disease Detection")

    # Image uploader for chest X-ray
    uploaded_chest_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "png", "jpeg"], key='chest')

    if uploaded_chest_file is not None:
        # Preprocess the image
        image = Image.open(uploaded_chest_file).convert('RGB')
        image = image.resize((150, 150))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict the chest disease type
        chest_prediction = chest_model.predict(image_array)
        chest_class = chest_classes[np.argmax(chest_prediction)]

        # Display the uploaded image
        st.image(uploaded_chest_file, caption='Uploaded Chest X-ray Image', use_column_width=True)
        st.write(f"**Predicted Disease:** {chest_class}")

        # Provide advice based on chest disease type
        if chest_class in chest_advice:
            st.write(f"### Advice for {chest_class}")
            st.write(f"**Doctors:** {chest_advice[chest_class]['Doctors']}")
            st.write(f"**Medication:** {chest_advice[chest_class]['Medication']}")
            st.write(f"**Surgery:** {chest_advice[chest_class]['Surgery']}")
            st.write(f"**Diets:** {chest_advice[chest_class]['Diets']}")

       
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #f8a5c2, #8A2BE2); /* Gradient from light pink to purple */
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
st.write("Made with ❤️ by HealthAI")
