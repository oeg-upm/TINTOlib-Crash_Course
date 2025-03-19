
from PIL import Image
import os

def zoom_image_nearest_neighbor(input_path, output_path, zoom_factor):
    # Open the image
    with Image.open(input_path) as img:
        # Get the original size
        width, height = img.size
        
        # Calculate the new size
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        
        # Resize the image using nearest neighbor interpolation
        resized_img = img.resize((new_width, new_height), Image.NEAREST)
        
        # Save the resized image
        resized_img.save(output_path)

def process_directory(input_dir, output_dir, zoom_factor):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{filename}")
            
            zoom_image_nearest_neighbor(input_path, output_path, zoom_factor)
            print(f"Processed: {filename}")