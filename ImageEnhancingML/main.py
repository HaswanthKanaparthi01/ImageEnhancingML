# Import necessary libraries
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# -----------------------------------
# 1. Functions to Extract Image Parameters
# -----------------------------------

model_path=r"\mobilenet_iter_73000.caffemodel" # use the path of these files 
config_path=r"\deploy.prototxt"

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

def crop_and_add_border(new_images_folder=None,enhanced_folder=None,execute_all=False,image_all=None):
    ############# Config ##############
    base_border_thickness=10
    confidence_threshold=0.5
    expansion_factor=0.05
    ############# Config ##############
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)
    if not execute_all:
        files = [f for f in os.listdir(new_images_folder) if os.path.isfile(os.path.join(new_images_folder, f))]
        for file in files:
            image_path = os.path.join(new_images_folder, file)
            print(f"Processing crop and add border: {image_path}")
            image = cv2.imread(image_path)
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            minX, minY = w, h
            maxX, maxY = 0, 0
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold:
                    idx = int(detections[0, 0, i, 1])
                    if idx == 15:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        minX = min(minX, startX)
                        minY = min(minY, startY)
                        maxX = max(maxX, endX)
                        maxY = max(maxY, endY)
            if minX < maxX and minY < maxY:
                width = maxX - minX
                height = maxY - minY
                expand_width = int(width * expansion_factor)
                expand_height = int(height * expansion_factor)
                minX = max(minX - expand_width, 0)
                minY = max(minY - expand_height, 0)
                maxX = min(maxX + expand_width, w)
                maxY = min(maxY + expand_height, h)
                border_thickness = int(base_border_thickness * max(w, h) / 1000)  # Adjust multiplier as needed
                cropped_image = image[minY:maxY, minX:maxX]
                bordered_image = cv2.copyMakeBorder(
                    cropped_image, 
                    border_thickness, 
                    border_thickness, 
                    border_thickness, 
                    border_thickness, 
                    cv2.BORDER_CONSTANT, 
                    value=[255, 255, 255]
                )
                output_file_path = os.path.join(enhanced_folder, file)
                # cv2.imshow("bordered_image",bordered_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(output_file_path, bordered_image)
                # os.remove(image_path)
                print(f"Processed image saved to {output_file_path}")
            else:
                border_thickness = int(base_border_thickness * max(w, h) / 1000)  # Adjust multiplier as needed
                bordered_image = cv2.copyMakeBorder(
                    image, 
                    border_thickness, 
                    border_thickness, 
                    border_thickness, 
                    border_thickness, 
                    cv2.BORDER_CONSTANT, 
                    value=[255, 255, 255]
                )
                output_file_path = os.path.join(enhanced_folder, file)
                cv2.imwrite(output_file_path, bordered_image)
                # os.remove(image_path)
                print(f"Processed image saved to {output_file_path}")
    else:
        image = image_all
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        minX, minY = w, h
        maxX, maxY = 0, 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # Consider only high confidence detections
            if confidence > confidence_threshold:
                idx = int(detections[0, 0, i, 1])
                if idx == 15:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    minX = min(minX, startX)
                    minY = min(minY, startY)
                    maxX = max(maxX, endX)
                    maxY = max(maxY, endY)
        if minX < maxX and minY < maxY:
            width = maxX - minX
            height = maxY - minY
            expand_width = int(width * expansion_factor)
            expand_height = int(height * expansion_factor)
            minX = max(minX - expand_width, 0)
            minY = max(minY - expand_height, 0)
            maxX = min(maxX + expand_width, w)
            maxY = min(maxY + expand_height, h)
            border_thickness = int(base_border_thickness * max(w, h) / 1000)  # Adjust multiplier as needed
            cropped_image = image[minY:maxY, minX:maxX]
            bordered_image = cv2.copyMakeBorder(
                cropped_image, 
                border_thickness, 
                border_thickness, 
                border_thickness, 
                border_thickness, 
                cv2.BORDER_CONSTANT, 
                value=[255, 255, 255]
            )
            return bordered_image
        else:
            border_thickness = int(base_border_thickness * max(w, h) / 1000)  # Adjust multiplier as needed
            bordered_image = cv2.copyMakeBorder(
                image, 
                border_thickness, 
                border_thickness, 
                border_thickness, 
                border_thickness, 
                cv2.BORDER_CONSTANT, 
                value=[255, 255, 255]
            )
            return bordered_image
# -----------------------------------
# 2. Load CSV Files and Train Models
# -----------------------------------

# Define the parameters
parameters = ['Brightness', 'Contrast', 'Exposure', 'Shadow', 'Tint']
input_columns = [f'Input_{param}' for param in parameters]
output_column = 'Output'

# Paths to the CSV files
csv_folder = r"C:\Users\kanaparthihaswanth\Desktop\Clients\Image_Enhancing_Clint"  

# Dictionary to store models for each parameter
models = {}

for param in parameters:
    # Load the CSV file for the parameter
    csv_file = os.path.join(csv_folder, f'{param.lower()}.csv')
    data = pd.read_csv(csv_file)
    
    # Prepare the data
    X = data[input_columns].values  # Input parameters
    y = data[output_column].values  # Output parameter
    
    # Split the data (optional, for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model (optional)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{param} Model Mean Squared Error: {mse:.4f}")
    
    # Store the trained model
    models[param] = model

print("All models have been trained successfully.")

# -----------------------------------
# 3. Enhancing New Images Using the Trained Models
# -----------------------------------

def adjust_image_parameters(image, adjustments):
    """Apply adjustments to image parameters."""
    # Unpack adjustments
    brightness_adj, contrast_adj, exposure_adj, shadow_adj, tint_adj = adjustments

    # Adjust brightness and contrast
    alpha = 1 + (contrast_adj / 128.0)  # Contrast adjustment factor
    beta = brightness_adj               # Brightness adjustment value
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Adjust exposure
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = hsv_image.astype(np.float32)
    hsv_image[:, :, 2] += exposure_adj
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2], 0, 255)
    hsv_image = hsv_image.astype(np.uint8)
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Adjust shadow using gamma correction
    gamma = 1 + shadow_adj  # Adjust gamma based on shadow adjustment
    if gamma != 0:
        invGamma = 1.0 / gamma
    else:
        invGamma = 0.1
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    image = cv2.LUT(image, table)

    # Adjust tint (hue shift)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = hsv_image.astype(np.float32)
    hsv_image[:, :, 0] += tint_adj
    hsv_image[:, :, 0] = np.mod(hsv_image[:, :, 0], 180)  # Hue values range from 0 to 179
    hsv_image = hsv_image.astype(np.uint8)
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return image

# Paths to your new images and where to save enhanced images
new_images_folder = r"C:\Users\kanaparthihaswanth\Desktop\Clients\Image_Enhancing_Clint\Before_images"         
enhanced_folder = r"C:\Users\kanaparthihaswanth\Desktop\Clients\Image_Enhancing_Clint\After_images_crop"  

if not os.path.exists(enhanced_folder):
    os.makedirs(enhanced_folder)

# Iterate over images in the new images folder
new_images = sorted(os.listdir(new_images_folder))

for img_name in new_images:
    img_path = os.path.join(new_images_folder, img_name)

    try:
        # Extract input parameters from the new image
        input_params = extract_image_parameters(img_path)
        input_params_array = np.array(input_params).reshape(1, -1)  # Reshape for prediction

        # Initialize list to store predicted output parameters
        predicted_outputs = []

        # Predict output parameters using the models
        for param in parameters:
            model = models[param]
            predicted_output = model.predict(input_params_array)[0]
            predicted_outputs.append(predicted_output)

        # Calculate adjustments (Output - Input)
        adjustments = np.array(predicted_outputs) - np.array(input_params)

        # Read the image
        image = cv2.imread(img_path)

        image=crop_and_add_border(execute_all=True,image_all=image)

        # Apply adjustments
        enhanced_image = adjust_image_parameters(image, adjustments)

        # Save the enhanced image
        save_path = os.path.join(enhanced_folder, img_name)
        cv2.imwrite(save_path, enhanced_image)
        print(f"Enhanced image saved as '{save_path}'")
    except Exception as e:
        print(f"Error processing {img_name}: {e}")

print("All images have been enhanced and saved.")
