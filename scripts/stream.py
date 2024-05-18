import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

# Load the trained YOLO model
model = YOLO("best.pt")

def plot_image_with_bboxes(image, bboxes, labels):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    for bbox, label in zip(bboxes, labels):
        x_center, y_center, width, height = bbox
        x_center, y_center, width, height = x_center * image.width, y_center * image.height, width * image.width, height * image.height
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min, label, color='yellow', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    
    st.pyplot(fig)

def main():
    st.title("YOLOv8 Object Detection")
    
    st.sidebar.title("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Running object detection...")
        
        results = model(image)
        
        bboxes = results[0].boxes.xywh.numpy()
        labels = [model.names[int(c)] for c in results[0].boxes.cls.numpy()]
        
        plot_image_with_bboxes(image, bboxes, labels)

if __name__ == "__main__":
    main()
