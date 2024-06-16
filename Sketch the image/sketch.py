import streamlit as st
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
 
def sketch_transform(image):
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (7,7), 0)
    image_canny = cv2.Canny(image_grayscale_blurred, 10, 80)
    _, mask = image_canny_inverted = cv2.threshold(image_canny, 30, 255, cv2.THRESH_BINARY_INV)
    return mask



def main():
    st.set_page_config(page_title="Sketch from Image", page_icon="🎨", layout="wide")
    st.title("Sketch from Image ")

    # Upload image file on the left side
    st.sidebar.title("Upload Image")
    uploaded_file = st.sidebar.file_uploader("", type=['png', 'jpg'])

    if uploaded_file is not None:
        # Convert the file to an opencv image
        image = np.array(Image.open(uploaded_file))
        
        # Display original image on the left side
        st.sidebar.subheader("Original Image")
        st.sidebar.image(image, caption="Original Image", use_column_width=True)
        
        # Process image
        detected_img = sketch_transform(image)
        
        # Resize the image to fit the frame
        resized_img = cv2.resize(detected_img, (280, 190))
        
        # Display processed image
        # Display processed image
        st.subheader("Processed  Image")
        st.image(resized_img,  caption="Image Sketching", use_column_width=True)
        
if __name__ == "__main__":
    main()