import pandas as pd
import os
from PIL import Image
import io

# Load the Parquet file
parquet_file = "huggin-sidewalk-imgs.parquet"  # Change this to your file path
output_folder = "output_images"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the Parquet file
df = pd.read_parquet(parquet_file)

# Assuming the image bytes are stored in a column named 'image_data'
for index, row in df.iterrows():
    image_bytes = row["image_data"]  # Adjust this column name if different
    
    # Convert bytes to image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Save the image
    output_path = os.path.join(output_folder, f"image_{index}.png")  # Change to .jpg if needed
    image.save(output_path, format="PNG")  # Change to "JPEG" for JPG format

    print(f"Saved: {output_path}")

print("Conversion complete!")
