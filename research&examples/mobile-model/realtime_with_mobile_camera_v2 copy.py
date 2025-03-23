import cv2
import numpy as np
import requests
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from matplotlib import pyplot as plt
from PIL import Image

url = "http://192.168.224.75:8080/shot.jpg"  # Replace with your phone's IP and port

# Initialize the segmentation model
model_name = "tobiasc/segformer-b0-finetuned-segments-sidewalk"
processor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
model.eval()  # Set to evaluation mode

# Class IDs for relevant objects
class_colors = {
    0: [0, 0, 0],       # Background - Black
    1: [128, 64, 128],   # Road - Purple-blue
    2: [244, 35, 232],   # Sidewalk - Pink
    3: [220, 220, 0],    # Crosswalk - Yellow
    4: [70, 70, 70],     # Building - Dark gray
    5: [102, 102, 156],  # Pole - Light purple
    6: [190, 153, 153],  # Traffic Light - Pinkish gray
    7: [153, 153, 153],  # Other - Gray
}

def analyze_frame(frame):
    """
    Analyze a camera frame with the segmentation model.
    
    Args:
        frame: Input frame from the camera.
    
    Returns:
        seg_map: Segmentation map with class IDs.
    """
    # Prepare image for the model
    inputs = processor(images=frame, return_tensors="pt")

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits and resize to original image size
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(frame.shape[0], frame.shape[1]),
        mode="bilinear",
        align_corners=False
    )

    # Get segmentation mask
    seg_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    return seg_map

def create_visualization(frame, segmentation_map):
    """
    Create a visualization of the segmentation map.
    
    Args:
        frame: Original camera frame.
        segmentation_map: Segmentation map with class IDs.
    
    Returns:
        blended: Frame with segmentation overlay.
    """
    # Create colored segmentation overlay
    colored_segmentation = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
    
    for class_id, color in class_colors.items():
        colored_segmentation[segmentation_map == class_id] = color
        
    # Blend with original frame
    alpha = 0.5
    blended = cv2.addWeighted(frame, 1-alpha, colored_segmentation, alpha, 0)
    
    return blended

# Main loop to fetch frames and process them
try:
    while True:
        # Fetch frame from the IP camera
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Analyze frame with the segmentation model
        segmentation_map = analyze_frame(frame)

        # Create visualization
        viz_frame = create_visualization(frame, segmentation_map)

        # Convert BGR to RGB for displaying with matplotlib
        viz_frame_rgb = cv2.cvtColor(viz_frame, cv2.COLOR_BGR2RGB)

        # Display the visualization using matplotlib
        plt.imshow(viz_frame_rgb)
        plt.axis('off')  # Hide axes
        plt.show()

        # Wait for a short time before fetching the next frame
        plt.pause(0.1)  # Adjust the delay as needed

except KeyboardInterrupt:
    print("Stopping the program.")

# Release resources
plt.close()