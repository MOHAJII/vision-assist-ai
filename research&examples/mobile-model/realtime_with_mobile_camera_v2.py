import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class SemanticSegmentation:
    def __init__(self, model_name="tobiasc/segformer-b0-finetuned-segments-sidewalk", camera_url="http://192.168.x.x:8080/video"):
        """
        Initialize the semantic segmentation model and camera.
        
        Args:
            model_name: Hugging Face model identifier or local path.
            camera_url: URL or index of the camera feed (e.g., IP Webcam URL).
        """
        # Load the segmentation model and processor
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode

        # Class IDs for relevant objects
        self.sidewalk_id = 2  # Class ID for sidewalk
        self.road_id = 1      # Class ID for road
        self.crosswalk_id = 3  # Class ID for crosswalk

        # Visualization settings
        self.viz_enabled = True
        self.class_colors = {
            0: [0, 0, 0],       # Background - Black
            1: [128, 64, 128],   # Road - Purple-blue
            2: [244, 35, 232],   # Sidewalk - Pink
            3: [220, 220, 0],    # Crosswalk - Yellow
            4: [70, 70, 70],     # Building - Dark gray
            5: [102, 102, 156],  # Pole - Light purple
            6: [190, 153, 153],  # Traffic Light - Pinkish gray
            7: [153, 153, 153],  # Other - Gray
        }

        # Initialize camera
        self.camera = cv2.VideoCapture(camera_url)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera feed: {camera_url}")

        # Initialize visualization window
        if self.viz_enabled:
            cv2.namedWindow("Semantic Segmentation", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Semantic Segmentation", 1280, 720)

    def analyze_frame(self, frame):
        """
        Analyze a camera frame with the segmentation model.
        
        Args:
            frame: Input frame from the camera.
        
        Returns:
            seg_map: Segmentation map with class IDs.
        """
        # Prepare image for the model
        inputs = self.processor(images=frame, return_tensors="pt")

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

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

    def create_visualization(self, frame, segmentation_map):
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
        
        for class_id, color in self.class_colors.items():
            colored_segmentation[segmentation_map == class_id] = color
            
        # Blend with original frame
        alpha = 0.5
        blended = cv2.addWeighted(frame, 1-alpha, colored_segmentation, alpha, 0)
        
        return blended

    def run(self):
        """
        Main loop for real-time semantic segmentation.
        """
        try:
            while True:
                # Capture frame-by-frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                # Resize for faster processing
                frame = cv2.resize(frame, (640, 480))

                # Analyze frame
                segmentation_map = self.analyze_frame(frame)

                # Create visualization
                if self.viz_enabled:
                    viz_frame = self.create_visualization(frame, segmentation_map)
                    cv2.imshow("Semantic Segmentation", viz_frame)

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Release the camera and close windows
            self.camera.release()
            cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Replace with your mobile phone's IP camera URL
    camera_url = "http://192.168.224.30:8080/video"  # Replace with your phone's IP and port

    # Initialize and run the semantic segmentation model
    segmenter = SemanticSegmentation(camera_url=camera_url)
    segmenter.run()