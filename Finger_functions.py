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









