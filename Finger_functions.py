import os
import random
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2


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

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #img = cv2.resize(img, (96, 103))
    img = img / 255.0
    return img