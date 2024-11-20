import os
import random
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
import numpy as np


path_real = './SOCOFing/Real/'
path_altered_easily = './SOCOFing/Altered/Altered-Easy/'
path_altered_medium = './SOCOFing/Altered/Altered-Medium/'
path_altered_hard = './SOCOFing/Altered/Altered-Hard/'

def get_path(finger, modification = 'real'):
    if modification == 'real':
        return path_real + finger
    if modification == 'easily':
        return path_altered_easily + finger
    if modification == 'medium':
        return path_altered_medium + finger
    if modification == 'hard':
        return path_altered_hard


def choose_random_finger(modification='real'):
    if modification == 'real':
        directory = path_real
    elif modification == 'easily':
        directory = path_altered_easily
    elif modification == 'medium':
        directory = path_altered_medium
    elif modification == 'hard':
        directory = path_altered_hard
    else:
        raise ValueError("Invalid modification level")

    files = os.listdir(directory)
    random_finger = random.choice(files)
    return get_path(random_finger, modification)


import os

def get_all_fingerprints_with_labels():
    real_files = [(os.path.join(path_real, file), 'real') for file in os.listdir(path_real)]
    easy_files = [(os.path.join(path_altered_easily, file), 'easily') for file in os.listdir(path_altered_easily)]
    medium_files = [(os.path.join(path_altered_medium, file), 'medium') for file in os.listdir(path_altered_medium)]
    hard_files = [(os.path.join(path_altered_hard, file), 'hard') for file in os.listdir(path_altered_hard)]

    all_files = real_files + easy_files + medium_files + hard_files

    def extract_index(file_tuple):
        filename = os.path.basename(file_tuple[0])
        index = int(filename.split('__')[0])
        return index

    all_files_sorted = sorted(all_files, key=extract_index)

    return all_files_sorted


def train_val_test_split(df, test_size=0.15, val_size=0.15):
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    val_relative_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=val_relative_size, random_state=42)

    return train_df, val_df, test_df


def extract_information_from_filename(file_path):
    filename = os.path.basename(file_path)
    filename = filename.split('.')[0]
    parts = filename.split('__')
    id = parts[0]
    if len(parts) != 2:
        raise ValueError(f"Filename '{filename}' is not in the expected format.")

    gender_hand_finger = parts[1].split('_')

    if len(gender_hand_finger) == 4:
        gender = 'Male' if gender_hand_finger[0] == 'M' else 'Female'
        hand = gender_hand_finger[1]
        finger = gender_hand_finger[2]
        method = None

    elif len(gender_hand_finger) == 5:
        gender = 'Male' if gender_hand_finger[0] == 'M' else 'Female'
        hand = gender_hand_finger[1]
        finger = gender_hand_finger[2]
        method = gender_hand_finger[4]

    return id, gender, hand, finger, method, file_path


def create_fingerprint_dataframe():
    all_files_with_labels = get_all_fingerprints_with_labels()
    data = []

    for file_path, modification in all_files_with_labels:
        id, gender, hand, finger, method, file_path = extract_information_from_filename(file_path)
        data.append([file_path, id, modification, gender, hand, finger, method])

    df = pd.DataFrame(data, columns=['file_path', 'id', 'modification', 'gender', 'hand', 'finger', 'method'])

    return df

def remove_frame(image_path):
    image = cv2.imread(image_path)
    x = 2
    y = 2
    w = 90
    h = 97
    x_end = min(x + w, 112)
    y_end = min(y + h, 112)
    cropped_image = image[y:y_end, x:x_end]
    return cropped_image

def remove_frame_from_array(image_array):
    x = 2
    y = 2
    w = 90
    h = 97
    x_end = min(x + w, image_array.shape[1])
    y_end = min(y + h, image_array.shape[0])
    cropped_image = image_array[y:y_end, x:x_end]
    return cropped_image


def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = remove_frame_from_array(img)
    img = cv2.resize(img, (112, 112))
    img = img / 255.0
    return img

def rotate_circle_region(image_array, min_radius=10, max_radius=25):
    """
    Selects a random circular region on the image array, rotates the region by a random degree,
    and returns the modified image array. Ensures the circle fits within the image boundaries.

    Args:
    - image_array (numpy array): Input image array.
    - min_radius (int): Minimum radius of the circle.
    - max_radius (int): Maximum radius of the circle.

    Returns:
    - numpy array: Modified image array with rotated circular region.
    """
    h, w = image_array.shape[:2]

    # Ensure the circle fits within the image boundaries
    max_radius = min(max_radius, w // 2, h // 2)

    # Randomly select a radius ensuring the circle fits in the selected center position
    radius = random.randint(min_radius, max_radius)

    # Randomly select a center point, ensuring the circle fits within the image boundaries
    center_x = random.randint(radius, w - radius)
    center_y = random.randint(radius, h - radius)

    # Generate a random degree of rotation
    angle = random.uniform(25, 335)

    # Create a circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, (255), thickness=-1)

    # Extract the circular region using the mask
    circular_region = cv2.bitwise_and(image_array, image_array, mask=mask)

    # Crop the bounding box around the circular region
    x1, y1 = max(0, center_x - radius), max(0, center_y - radius)
    x2, y2 = min(w, center_x + radius), min(h, center_y + radius)
    cropped_circle = circular_region[y1:y2, x1:x2]

    # Create a mask for the cropped circle region
    circle_mask = mask[y1:y2, x1:x2]

    # Rotate the cropped circular region without black borders
    rotation_matrix = cv2.getRotationMatrix2D((radius, radius), angle, scale=1)
    rotated_circle = cv2.warpAffine(
        cropped_circle, rotation_matrix, (cropped_circle.shape[1], cropped_circle.shape[0]),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )

    # Reapply the circular mask to keep only the circle
    rotated_circle_masked = cv2.bitwise_and(rotated_circle, rotated_circle, mask=circle_mask)

    # Insert the rotated circle back into the original image
    result_image = image_array.copy()
    for i in range(rotated_circle_masked.shape[0]):
        for j in range(rotated_circle_masked.shape[1]):
            if circle_mask[i, j] != 0:  # Update only within the circular region
                result_image[y1 + i, x1 + j] = rotated_circle_masked[i, j]

    return result_image

def generate_modified_fingerprint_dataframe(fingerprint_df, image_column, n_rows):
    """
    Generates a modified DataFrame with modified fingerprint images,
    only modifying rows where the "modification" column is 0. Ensures the required
    number of modified images is returned.

    Args:
    - fingerprint_df (pd.DataFrame): Original DataFrame containing fingerprint data.
    - image_column (str): Column name containing the image arrays.
    - n_rows (int): Number of rows to process from the original DataFrame.

    Returns:
    - pd.DataFrame: New DataFrame with modified fingerprint data.
    """
    modified_data = []

    # Filter rows with modification == 0
    eligible_rows = fingerprint_df[fingerprint_df["modification"] == 0]

    if eligible_rows.empty:
        raise ValueError("No non-modified rows available for processing.")

    # Ensure we get exactly n_rows modified images
    processed_count = 0
    for index, row in eligible_rows.iterrows():
        if processed_count >= n_rows:
            break  # Stop once we have the required number of modified images

        # Retrieve the specific image for the current row
        original_image = row[image_column]

        # Modify the fingerprint image uniquely for this row
        modified_image = rotate_circle_region(original_image)

        # Create a new row for the modified fingerprint
        new_row = {
            "file_path": "",  # Empty for modified fingerprints
            "id": row["id"],  # Use the original ID
            "modification": 4,  # Indicate this is a modified fingerprint
            "gender": row["gender"],  # Copy gender from original
            "hand": row["hand"],  # Copy hand from original
            "finger": row["finger"],  # Copy finger from original
            "method": 4,  # Indicate the modification method
            "image_data": modified_image  # Add the modified image
        }
        modified_data.append(new_row)
        processed_count += 1

    # Check if we reached the required count
    if processed_count < n_rows:
        raise ValueError(
            f"Only {processed_count} non-modified rows available for processing; {n_rows} required."
        )

    # Create a DataFrame from the modified data
    modified_df = pd.DataFrame(modified_data)

    return modified_df


def obliterate_circle_region_with_noise(image_array, min_radius=5, max_radius=20):
    """
    Obliterates (distorts) a random circular region on the image with noise or smudging.

    Args:
    - image_array (numpy array): Input image array.
    - min_radius (int): Minimum radius of the obliterated circle.
    - max_radius (int): Maximum radius of the obliterated circle.

    Returns:
    - numpy array: Image array with the obliterated circular region.
    """
    # Get the dimensions of the image
    h, w = image_array.shape[:2]

    # Ensure the circle fits within the image boundaries
    max_radius = min(max_radius, w // 2, h // 2)

    # Randomly select a radius ensuring the circle fits in the selected center position
    radius = random.randint(min_radius, max_radius)

    # Randomly select a center point, ensuring the circle fits within the image boundaries
    center_x = random.randint(radius, w - radius)
    center_y = random.randint(radius, h - radius)

    # Create a circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, thickness=-1)

    # Extract the circular region
    obliterated_image = image_array.copy()
    circular_region = cv2.bitwise_and(obliterated_image, obliterated_image, mask=mask)

    # Define the bounding box for the circle
    x1, y1 = max(0, center_x - radius), max(0, center_y - radius)
    x2, y2 = min(w, center_x + radius), min(h, center_y + radius)
    cropped_circle = circular_region[y1:y2, x1:x2]

    # Apply a random blur, noise, or distortion effect
    effect_choice = random.choice(["blur", "pixelate"])
    if effect_choice == "blur":
        # Heterogeneous blur
        obliterated_effect = np.copy(cropped_circle)
        # Randomize the kernel size for the entire circular region
        random_kernel = random.choice([3, 5, 7, 9])  # Randomize kernel size for the blur
        obliterated_effect = cv2.GaussianBlur(cropped_circle, (random_kernel, random_kernel), 0)
    #elif effect_choice == "noise":
    #    # Add random noise
    #    noise = np.random.normal(0, 25, cropped_circle.shape).astype(np.int16)  # Use int16 to avoid overflow
    #    cropped_circle_int16 = cropped_circle.astype(np.int16)  # Convert cropped_circle to int16
    #    obliterated_effect = np.clip(cropped_circle_int16 + noise, 0, 255).astype(np.uint8)  # Add noise and clip
    elif effect_choice == "pixelate":
        # Downscale and upscale to create a pixelated effect
        downscale_size = (max(1, cropped_circle.shape[1] // 5), max(1, cropped_circle.shape[0] // 5))
        pixelated = cv2.resize(cropped_circle, downscale_size, interpolation=cv2.INTER_LINEAR)
        obliterated_effect = cv2.resize(pixelated, (cropped_circle.shape[1], cropped_circle.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Place the obliterated effect back into the circular region
    for i in range(obliterated_effect.shape[0]):
        for j in range(obliterated_effect.shape[1]):
            if mask[y1 + i, x1 + j] == 255:  # Apply only within the circular mask
                obliterated_image[y1 + i, x1 + j] = obliterated_effect[i, j]

    return obliterated_image

def generate_modified_fingerprint_dataframe_obliteration(fingerprint_df, image_column, n_rows):
    """
    Generates a modified DataFrame with modified fingerprint images,
    only modifying rows where the "modification" column is 0. Ensures the required
    number of modified images is returned.

    Args:
    - fingerprint_df (pd.DataFrame): Original DataFrame containing fingerprint data.
    - image_column (str): Column name containing the image arrays.
    - n_rows (int): Number of rows to process from the original DataFrame.

    Returns:
    - pd.DataFrame: New DataFrame with modified fingerprint data.
    """
    modified_data = []

    # Filter rows with modification == 0
    eligible_rows = fingerprint_df[fingerprint_df["modification"] == 0]

    if eligible_rows.empty:
        raise ValueError("No non-modified rows available for processing.")

    # Ensure we get exactly n_rows modified images
    processed_count = 0
    for index, row in eligible_rows.iterrows():
        if processed_count >= n_rows:
            break  # Stop once we have the required number of modified images

        # Retrieve the specific image for the current row
        original_image = row[image_column]

        # Modify the fingerprint image uniquely for this row
        modified_image = obliterate_circle_region_with_noise(original_image)

        # Create a new row for the modified fingerprint
        new_row = {
            "file_path": "",  # Empty for modified fingerprints
            "id": row["id"],  # Use the original ID
            "modification": 4,  # Indicate this is a modified fingerprint
            "gender": row["gender"],  # Copy gender from original
            "hand": row["hand"],  # Copy hand from original
            "finger": row["finger"],  # Copy finger from original
            "method": 4,  # Indicate the modification method
            "image_data": modified_image  # Add the modified image
        }
        modified_data.append(new_row)
        processed_count += 1

    # Check if we reached the required count
    if processed_count < n_rows:
        raise ValueError(
            f"Only {processed_count} non-modified rows available for processing; {n_rows} required."
        )

    # Create a DataFrame from the modified data
    modified_df = pd.DataFrame(modified_data)

    return modified_df



