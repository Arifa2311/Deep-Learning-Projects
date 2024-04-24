import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import keras
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2

from keras.models import Sequential, Model
from keras.layers import InputLayer, Conv2D, Dense, Flatten, MaxPooling2D
from sklearn.preprocessing import LabelEncoder

# Load labels
labels = pd.read_json(r"C:\Users\Afrin\Downloads\data\data\class_dict_10.json")
label_encoder = LabelEncoder()

# Function to load images and labels
def load_data():
    fv = []
    cv = []
    for folder in os.listdir(r"C:\Users\Afrin\Downloads\data\data\class_10_train"):
        for files in os.listdir(r"C:\Users\Afrin\Downloads\data\data\class_10_train\{}\images".format(folder)):
            fv.append(cv2.imread(r"C:\Users\Afrin\Downloads\data\data\class_10_train\{}\images\{}".format(folder, files)))
            cv.append(labels.loc["class"][folder])
    ffv = np.asarray(fv)
    le = LabelEncoder()
    fcv = le.fit_transform(cv)
    return ffv, fcv

# Function to define CNN model
def create_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(64, 64, 3)))
    model.add(Conv2D(filters=10, kernel_size=(3, 3), activation="relu", strides=(1, 1), padding="valid"))
    model.add(Conv2D(filters=10, kernel_size=(3, 3), activation="relu", strides=(1, 1), padding="valid"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(Conv2D(filters=10, kernel_size=(3, 3), activation="relu", strides=(1, 1), padding="valid"))
    model.add(Conv2D(filters=10, kernel_size=(3, 3), activation="relu", strides=(1, 1), padding="valid"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(Flatten())
    model.add(Dense(units=10, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Function to extract feature maps
def extract_feature_maps(model, image):
    layer_outputs = [layer.output for layer in model.layers[:5]]
    activation_model = Model(inputs=model.inputs, outputs=layer_outputs)
    return activation_model.predict(image[np.newaxis])

# Main function to run Streamlit app
def main(model):
    st.title('Image Classification and Feature Extraction')
    st.sidebar.title('Options')
    selected_image = st.sidebar.file_uploader('Upload an image')

    if selected_image is not None:
        st.image(selected_image, caption='Uploaded Image', use_column_width=True)
        img = cv2.imdecode(np.fromstring(selected_image.read(), np.uint8), 1)
        
        # Load labels
        labels = pd.read_json(r"C:\Users\Afrin\Downloads\data\data\class_dict_10.json")
        label_encoder = LabelEncoder()
        
        # Fit LabelEncoder
        label_encoder.fit(labels.loc["class"].values)
        
        st.write("Predicted Label:", label_encoder.inverse_transform([np.argmax(model.predict(img[np.newaxis]))])[0])
        st.write("Extracted Feature Maps:")
        # Call the model with some dummy input data
        model.predict(np.zeros((1, 64, 64, 3)))
        feature_maps = extract_feature_maps(model, img)
        for z in [0, 1, 3, 4]:
            st.subheader(f'Convolutional Layer {z+1} Output')
            for i in range(1, 11):
                st.image(feature_maps[z][0][:, :, i - 1], caption=f'Feature Map {i}', use_column_width=True, clamp=True)

if __name__ == '__main__':
    ffv, fcv = load_data()
    model = create_model()
    model.fit(ffv, fcv, batch_size=500, epochs=50, validation_split=0.2)
    main(model)