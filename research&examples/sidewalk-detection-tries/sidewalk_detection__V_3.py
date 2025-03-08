import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import cv2
import numpy as np
import time
import pyttsx3  # For text-to-speech
import geopy.distance
import requests
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class BlindAssistantNavigator:
    def __init__(self, model_name="tobiasc/segformer-b0-finetuned-segments-sidewalk"):
        # Initialize the model
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval()

        # Initialize object detection model for obstacles
        self.obstacle_detector = cv2.dnn.readNetFromDarknet(
            "/home/mohammed/Documents/Documents/ENSET_2023-2026_interne/S4/InnovationProject/vision-assist-ai/research&examples/sidewalk-detection-tries/yolov4-tiny.cfg", 
            "/home/mohammed/Documents/Documents/ENSET_2023-2026_interne/S4/InnovationProject/vision-assist-ai/research&examples/sidewalk-detection-tries/yolov4-tiny.weights"
        )
        
        # Load COCO class names for obstacle detection
        with open("/home/mohammed/Documents/Documents/ENSET_2023-2026_interne/S4/InnovationProject/vision-assist-ai/research&examples/sidewalk-detection-tries/coco.names", "r") as f:
            self.obstacle_classes = f.read().strip().split("\n")
        
        # Relevant obstacle classes to detect
        self.relevant_obstacles = ["person", "bicycle", "car", "motorcycle", "bus", 
                                  "truck", "fire hydrant", "stop sign", "bench", 
                                  "dog", "cat"]

        # Text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speaking rate

        # Class IDs for relevant objects
        self.sidewalk_id = 2  # Class ID for sidewalk
        self.road_id = 1      # Class ID for road
        self.crosswalk_id = 3 # Class ID for crosswalk

        # Navigation state
        self.destination = None
        self.on_sidewalk = False
        self.last_instruction_time = 0
        self.instruction_cooldown = 5  # Seconds between instructions
        
        # Crosswalk state
        self.crosswalk_detected = False
        self.at_crosswalk_boundary = False
        
        # Visualization
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
        
        # Create a colormap for visualization
        self.colormap = np.array([self.class_colors[i] for i in range(8)]) / 255.0
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        
        # Initialize visualization window
        if self.viz_enabled:
            cv2.namedWindow("Blind Assistant Visualization", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Blind Assistant Visualization", 1280, 720)

    def speak(self, message):
        """Speak a message to the user"""
        print(f"Assistant: {message}")
        self.tts_engine.say(message)
        self.tts_engine.runAndWait()

    def set_destination(self, destination_name):
        """Set the destination (e.g., 'pharmacy')"""
        # In a real app, this would use a geocoding API
        self.destination = destination_name
        self.speak(f"Setting destination to the nearest {destination_name}.")

    def get_direction_to_destination(self):
        """Get direction to destination using GPS and mapping API"""
        # Simplified - in real implementation would use GPS and mapping API
        # Returns bearing in degrees (0=North, 90=East, etc.)
        return 45  # Example: destination is northeast

    def get_current_heading(self):
        """Get current heading from device compass"""
        # Simplified - in real implementation would use device compass
        return 90  # Example: facing east

    def analyze_frame(self, frame):
        """Analyze a camera frame with the segmentation model"""
        # Prepare image for model
        inputs = self.processor(images=frame, return_tensors="pt")

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get prediction
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
    
    def detect_obstacles(self, frame):
        """Detect obstacles using YOLO"""
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.obstacle_detector.setInput(blob)
        
        # Get detection layers
        layer_names = self.obstacle_detector.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.obstacle_detector.getUnconnectedOutLayers()]
        
        # Forward pass
        detections = self.obstacle_detector.forward(output_layers)
        
        # Process detections
        obstacles = []
        
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5 and self.obstacle_classes[class_id] in self.relevant_obstacles:
                    # Get bounding box coordinates
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    
                    # Calculate positions
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    # Determine position relative to user
                    relative_x = "center"
                    if center_x < width * 0.33:
                        relative_x = "left"
                    elif center_x > width * 0.66:
                        relative_x = "right"
                    
                    relative_dist = "far"
                    if h > height * 0.3:
                        relative_dist = "close"
                    elif h > height * 0.15:
                        relative_dist = "medium distance"
                    
                    obstacles.append({
                        "class": self.obstacle_classes[class_id],
                        "position": relative_x,
                        "distance": relative_dist,
                        "box": (x, y, w, h),
                        "confidence": confidence
                    })
        
        return obstacles

    def detect_crosswalk(self, segmentation_map):
        """Detect if we're approaching a crosswalk"""
        height, width = segmentation_map.shape
        
        # Check bottom half of image for crosswalk
        bottom_half = segmentation_map[int(height*0.5):, :]
        crosswalk_pixels = np.sum(bottom_half == self.crosswalk_id)
        
        # If significant crosswalk pixels in bottom half, we're approaching one
        if crosswalk_pixels > (bottom_half.size * 0.1):
            self.crosswalk_detected = True
            
            # Check if we're at the boundary (beginning of crosswalk)
            bottom_row = segmentation_map[int(height*0.9):, :]
            crosswalk_at_feet = np.sum(bottom_row == self.crosswalk_id) > (bottom_row.size * 0.2)
            
            if crosswalk_at_feet:
                self.at_crosswalk_boundary = True
            else:
                self.at_crosswalk_boundary = False
                
            return True
        
        self.crosswalk_detected = False
        self.at_crosswalk_boundary = False
        return False

    def determine_navigation_instruction(self, segmentation_map, obstacles):
        """Determine navigation instruction based on sidewalk detection"""
        height, width = segmentation_map.shape
        
        # First priority: Immediate obstacles
        if obstacles:
            close_obstacles = [o for o in obstacles if o["distance"] == "close"]
            if close_obstacles:
                obstacle = close_obstacles[0]
                return f"Caution: {obstacle['class']} {obstacle['distance']} to your {obstacle['position']}."
        
        # Second priority: Crosswalk detection
        self.detect_crosswalk(segmentation_map)
        if self.crosswalk_detected:
            if self.at_crosswalk_boundary:
                return "You have reached a crosswalk. Check for traffic before crossing."
            else:
                return "Crosswalk ahead. Continue walking forward."
                
        # Third priority: Sidewalk guidance
        # Check if user is on sidewalk (center-bottom of image)
        bottom_center_region = segmentation_map[int(height*0.7):, int(width*0.3):int(width*0.7)]
        on_sidewalk = np.sum(bottom_center_region == self.sidewalk_id) > (bottom_center_region.size * 0.4)

        # Find sidewalk locations in different parts of image
        left_region = segmentation_map[:, :int(width*0.3)]
        right_region = segmentation_map[:, int(width*0.7):]

        sidewalk_left = np.sum(left_region == self.sidewalk_id) > (left_region.size * 0.2)
        sidewalk_right = np.sum(right_region == self.sidewalk_id) > (right_region.size * 0.2)

        # Check for obstacles in path
        medium_obstacles = [o for o in obstacles if o["distance"] in ["medium distance", "close"]]

        # Determine instruction
        if not on_sidewalk:
            if sidewalk_left:
                return "Not on sidewalk. Turn left to find sidewalk."
            elif sidewalk_right:
                return "Not on sidewalk. Turn right to find sidewalk."
            else:
                return "No sidewalk detected. Please be cautious and scan around."
        else:
            # We're on a sidewalk - now get directional guidance
            current_heading = self.get_current_heading()
            target_heading = self.get_direction_to_destination()

            # Calculate turn needed (simplified)
            heading_diff = (target_heading - current_heading) % 360

            # If there are obstacles, provide avoidance instructions
            if medium_obstacles:
                obstacle = medium_obstacles[0]
                if obstacle["position"] == "center":
                    if sidewalk_left and not any(o["position"] == "left" for o in medium_obstacles):
                        return f"{obstacle['class']} ahead. Move left to avoid."
                    elif sidewalk_right and not any(o["position"] == "right" for o in medium_obstacles):
                        return f"{obstacle['class']} ahead. Move right to avoid."
                    else:
                        return f"Stop. {obstacle['class']} blocking path."

            # Regular navigation if no obstacles
            if heading_diff < 20 or heading_diff > 340:
                return "Continue straight on the sidewalk."
            elif 20 <= heading_diff <= 160:
                return f"Turn right at the next opportunity to head toward the {self.destination}."
            else:
                return f"Turn left at the next opportunity to head toward the {self.destination}."

    def create_visualization(self, frame, segmentation_map, obstacles, instruction):
        """Create visualization of detected elements"""
        # Create colored segmentation overlay
        colored_segmentation = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
        
        for class_id, color in self.class_colors.items():
            colored_segmentation[segmentation_map == class_id] = color
            
        # Blend with original frame
        alpha = 0.5
        blended = cv2.addWeighted(frame, 1-alpha, colored_segmentation, alpha, 0)
        
        # Draw obstacles
        for obstacle in obstacles:
            x, y, w, h = obstacle["box"]
            cv2.rectangle(blended, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{obstacle['class']} ({obstacle['distance']})"
            cv2.putText(blended, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw current instruction
        # Split instruction into lines of max 40 chars for better readability
        words = instruction.split()
        lines = []
        current_line = words[0]
        
        for word in words[1:]:
            if len(current_line + " " + word) <= 40:
                current_line += " " + word
            else:
                lines.append(current_line)
                current_line = word
        
        lines.append(current_line)
        
        # Draw black background for text
        for i, line in enumerate(lines):
            cv2.rectangle(blended, (10, 30 + i*30 - 20), (10 + len(line)*11, 30 + i*30 + 10), (0, 0, 0), -1)
            cv2.putText(blended, line, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw legend
        legend_y = 100
        width = 300
        cv2.putText(blended, "Legend:", (width - 150, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        legend_items = [
            ("Sidewalk", self.class_colors[self.sidewalk_id]),
            ("Road", self.class_colors[self.road_id]),
            ("Crosswalk", self.class_colors[self.crosswalk_id]),
        ]
        
        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y + 30 + i*25
            cv2.rectangle(blended, (width - 150, y_pos), (width - 130, y_pos + 20), color, -1)
            cv2.putText(blended, label, (width - 120, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return blended

    def run(self):
        """Main loop for the assistant"""
        self.speak("Blind navigation assistant started. I'll help you navigate to your destination safely.")
        self.set_destination("pharmacy")

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    self.speak("Camera error. Please restart the application.")
                    break

                # Resize for faster processing
                frame = cv2.resize(frame, (640, 480))
                height, width = frame.shape[:2]

                # Analyze frame
                segmentation_map = self.analyze_frame(frame)
                
                # Detect obstacles
                obstacles = self.detect_obstacles(frame)

                # Get navigation instruction
                instruction = self.determine_navigation_instruction(segmentation_map, obstacles)

                # Speak instruction if enough time has passed
                current_time = time.time()
                if current_time - self.last_instruction_time > self.instruction_cooldown:
                    self.speak(instruction)
                    self.last_instruction_time = current_time
                
                # Create and show visualization
                if self.viz_enabled:
                    viz_frame = self.create_visualization(frame, segmentation_map, obstacles, instruction)
                    cv2.imshow("Blind Assistant Visualization", viz_frame)

                # Check for stop command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('v'):  # Toggle visualization
                    self.viz_enabled = not self.viz_enabled

        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            self.speak("Navigation assistant stopped.")

# Example usage
if __name__ == "__main__":
    navigator = BlindAssistantNavigator()
    navigator.run()