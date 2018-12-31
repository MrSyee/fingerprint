import os
from PIL import Image

def get_image_paths(path):
    paths = list()
    for root, dirs, files in os.walk(path):
        for file_name in files:
            paths.append(os.path.join(root, file_name))

    return paths

def image_crop(paths, crop_area):
    for path in paths:
        result = Image.open(path).crop(crop_area)
        result.save(path)

crop_area = (0, 0, 114, 57)

if __name__ == '__main__':
    with open("data_path.txt", 'r') as file:
        file_list = list(file)
        TRAIN_DATASET_PATH = file_list[0].strip()
        TEST_DATASET_PATH = file_list[1].strip()

    paths = get_image_paths(TRAIN_DATASET_PATH)
    image_crop(paths, crop_area)
    paths = get_image_paths(TEST_DATASET_PATH)
    image_crop(paths, crop_area)
