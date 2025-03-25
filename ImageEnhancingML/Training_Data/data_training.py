# Import necessary libraries
import os
import cv2
import numpy as np
import pandas as pd

# -----------------------------------
# 1. Functions to Extract Image Parameters
# -----------------------------------

def calculate_brightness(image):
    """Calculate the brightness of an image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_image)
    return brightness

def calculate_contrast(image):
    """Calculate the contrast of an image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = gray_image.std()
    return contrast

def calculate_exposure(image):
    """Calculate the exposure of an image."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    exposure = np.mean(hsv_image[:, :, 2])  # V channel
    return exposure

def calculate_shadow(image):
    """Calculate the shadow ratio of an image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shadow_threshold = 50  # Threshold for shadow (0-255)
    shadow_pixels = np.sum(gray_image < shadow_threshold)
    total_pixels = gray_image.size
    shadow_ratio = shadow_pixels / total_pixels
    return shadow_ratio

def calculate_tint(image):
    """Calculate the tint (average hue) of an image."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tint = np.mean(hsv_image[:, :, 0])  # H channel
    return tint

def extract_image_parameters(image_path):
    """Extract parameters from an image."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    brightness = calculate_brightness(image)
    contrast = calculate_contrast(image)
    exposure = calculate_exposure(image)
    shadow = calculate_shadow(image)
    tint = calculate_tint(image)
    return [brightness, contrast, exposure, shadow, tint]

# -----------------------------------
# 2. Creating Individual CSV Files for Each Parameter
# -----------------------------------

# Initialize lists to store parameters
input_parameters_list = []
output_parameters_list = []

# Paths to your input and output image folders
input_folder = r" "    #give input images folder path
output_folder = r" "  #give expected images folder path

# Get lists of image filenames in input and output folders
input_images = os.listdir(input_folder)
output_images = os.listdir(output_folder)

# Convert lists to sets for efficient lookup
input_images_set = set(input_images)
output_images_set = set(output_images)

# Find common images by filename
common_images = input_images_set.intersection(output_images_set)

if not common_images:
    raise ValueError("No matching image filenames found between input and output folders.")

# Iterate over common images
for img_name in sorted(common_images):
    # Full paths
    input_img_path = os.path.join(input_folder, img_name)
    output_img_path = os.path.join(output_folder, img_name)
    
    try:
        # Extract parameters
        input_params = extract_image_parameters(input_img_path)
        output_params = extract_image_parameters(output_img_path)
        
        # Append data to lists
        input_parameters_list.append(input_params)
        output_parameters_list.append(output_params)
    except Exception as e:
        print(f"Error processing '{img_name}': {e}")

# Define parameter names
parameters = ['Brightness', 'Contrast', 'Exposure', 'Shadow', 'Tint']

# Create DataFrames for input and output parameters
input_df = pd.DataFrame(input_parameters_list, columns=[f'Input_{param}' for param in parameters])
output_df = pd.DataFrame(output_parameters_list, columns=[f'Output_{param}' for param in parameters])

# Combine input and output DataFrames
data_table = pd.concat([input_df, output_df], axis=1)

# Create individual CSV files for each parameter
for param in parameters:
    # Select the relevant columns
    df = data_table[[f'Input_Brightness', f'Input_Contrast', f'Input_Exposure', f'Input_Shadow', f'Input_Tint', f'Output_{param}']]
    # Rename Output column to 'Output'
    df = df.rename(columns={f'Output_{param}': 'Output'})
    # Save to CSV
    csv_filename = f'{param.lower()}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"CSV file '{csv_filename}' created.")

print("All CSV files have been created successfully.")
