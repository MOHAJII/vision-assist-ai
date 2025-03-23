import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyttsx3
from PIL import Image
import threading
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()
# Lower rate for faster speech
engine.setProperty('rate', 175)

# Global variables for threading
latest_instruction = "Initializing..."
speak_instruction = False

def tts_thread_function():
    """Thread function for text-to-speech to avoid blocking the main thread"""
    global latest_instruction, speak_instruction
    last_spoken = ""
    while True:
        if speak_instruction and latest_instruction != last_spoken:
            engine.say(latest_instruction)
            engine.runAndWait()
            last_spoken = latest_instruction
            speak_instruction = False
        time.sleep(0.1)

def test_segformer_realtime(model_name, ip_webcam_url):
    """
    Test the SegFormer model in real-time using a camera feed.

    Args:
        model_name: Hugging Face model identifier or local path
        ip_webcam_url: URL for IP Webcam (e.g., http://192.168.1.100:8080/video)
    """
    global latest_instruction, speak_instruction
    
    # Start TTS thread
    tts_thread = threading.Thread(target=tts_thread_function, daemon=True)
    tts_thread.start()
    
    # Set up CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and processor
    print(f"Loading model: {model_name}")
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Define class names and colors 
    class_names = {
        0: "unlabeled",
        1: "road",
        2: "sidewalk",
        3: "crosswalk",
        4: "cycling lane",
        5: "parking/driveway",
    }

    # Create a colormap (distinct colors for each class)
    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(30)]  # Supports up to 30 classes

    # Open the IP Webcam feed
    print(f"Connecting to IP Webcam at: {ip_webcam_url}")
    cap = cv2.VideoCapture(ip_webcam_url)
    
    if not cap.isOpened():
        print("Error: Could not open IP Webcam feed.")
        return

    # Set lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("IP Webcam connected successfully. Starting detection...")

    # Performance tracking
    frame_count = 0
    start_time = time.time()
    skip_frames = 0  # Used to process only every n-th frame
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
            
        frame_count += 1
        
        # Skip frames to increase speed
        if frame_count % 3 != 0:  # Process every 3rd frame
            # Just display the original frame with the last status text
            cv2.imshow('Real-Time Segmentation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
            
        # Resize for faster processing
        frame = cv2.resize(frame, (320, 240))
            
        # Convert the frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Prepare image for the model
        inputs = processor(images=image, return_tensors="pt")
        
        # Move inputs to the same device as model
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get logits and resize to original image size
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False
        )

        # Get prediction masks
        seg_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # Create a mask specifically for sidewalks (class_id 2)
        sidewalk_mask = (seg_map == 2)

        # Calculate percentage of sidewalk in the image
        sidewalk_percentage = np.sum(sidewalk_mask) / sidewalk_mask.size * 100

        # Determine if person is likely on sidewalk (simplified: check bottom center of image)
        h, w = seg_map.shape
        bottom_center = seg_map[int(0.8*h):, int(0.3*w):int(0.7*w)]
        on_sidewalk = np.sum(bottom_center == 2) > (bottom_center.size * 0.4)

        # 2. Sidewalk overlay on original
        img_array = np.array(image)
        
        # Create a colored mask for visualization (simple version for speed)
        mask_colored = np.zeros_like(img_array)
        mask_colored[sidewalk_mask] = [0, 255, 0]  # Green for sidewalk
        
        # Faster visualization with simple alpha blending
        alpha = 0.4
        combined_image = cv2.addWeighted(img_array, 1, mask_colored, alpha, 0)
        
        # Resize back to display size
        combined_image = cv2.resize(combined_image, (640, 480))

        # Add text showing if the person is on sidewalk and sidewalk percentage
        status_text = f"{'ON' if on_sidewalk else 'NOT on'} sidewalk ({sidewalk_percentage:.1f}%)"
        color = (0, 255, 0) if on_sidewalk else (0, 0, 255)
        cv2.putText(combined_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Add FPS information
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(combined_image, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the combined image
        cv2.imshow('Real-Time Segmentation', combined_image)

        # Generate navigation instructions
        if on_sidewalk:
            instruction = "Continue forward on the sidewalk."
        else:
            # Check left and right portions for sidewalk
            left_region = seg_map[:, :int(w*0.3)]
            right_region = seg_map[:, int(w*0.7):]

            sidewalk_left = np.sum(left_region == 2) > (left_region.size * 0.2)
            sidewalk_right = np.sum(right_region == 2) > (right_region.size * 0.2)

            if sidewalk_left:
                instruction = "Turn LEFT to reach the sidewalk."
            elif sidewalk_right:
                instruction = "Turn RIGHT to reach the sidewalk."
            else:
                instruction = "No sidewalk detected nearby. Scan the area by turning slowly."

        # Update instruction for TTS thread (speak every ~3 seconds)
        if frame_count % 15 == 0:  # Assuming ~5 FPS, this would be every 3 seconds
            latest_instruction = instruction
            speak_instruction = True

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Model name
    model_name = "tobiasc/segformer-b3-finetuned-segments-sidewalk"

    # IP Webcam URL - Replace with your phone's actual IP address
    # Make sure to add "/video" at the end to get the video stream
    ip_webcam_url = "http://192.168.224.30:8080/video"  # Replace with your phone's IP

    # Run the real-time test
    test_segformer_realtime(model_name, ip_webcam_url)