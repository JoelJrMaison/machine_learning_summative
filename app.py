# Save the current cell content to a file named 'app.py'

import streamlit as st
import torch
import numpy as np
from PIL import Image
from joel_maison_machine_learning_lane_line_detection_project import ENet, LaneDetector

@st.cache(allow_output_mutation=True)
def load_model():
    # Create an instance of the ENet model
    enet_model = ENet(2, 4)

    # Load the saved state dictionary into the model
    enet_model.load_state_dict(torch.load('lane_detection_model.pth', map_location=torch.device('cpu')))

    # Switch the model to evaluation mode
    enet_model.eval()

    # Create a LaneDetector instance using the ENet model
    lane_detector = LaneDetector(enet_model)

    return lane_detector

# Streamlit UI
def main():
    st.title("Lane Detection App")

    # File uploader for image selection
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Read and convert the uploaded image to a NumPy array
        image = Image.open(uploaded_file).convert('RGB')
        image = np.array(image)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Process the image on button click
        if st.button('Detect Lanes'):
            # Load the lane detection model
            lane_detector = load_model()

            try:
                # Process the image and detect lanes
                processed_image, lanes = lane_detector(image)

                # Display the processed image and the number of detected lanes
                st.image(processed_image, caption='Processed Image', use_column_width=True)
                st.write(f"Detected {len(lanes)} lanes")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
