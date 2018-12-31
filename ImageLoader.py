import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data


class ImageLoader(data.Dataset):
    def __init__(self, root, phase='', transforms=None):
        self.image_paths = self.get_image_paths(root, phase)
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        # label 1 (Real or Fake)
        if 'Real' in image_path:
            label = np.array(1)
        else:
            label = np.array(0)

        # image 1 x 57 x 116
        image = Image.open(image_path)
        # image = np.array(image, dtype=np.float32)
        # image = np.expand_dims(np.array(image, dtype=np.float32), axis=0)

        if self.transforms is not None:
            image = self.transforms(image)

        return (image, label)

    def __len__(self):
        # Enables you to get the length of the data by calling len(ImageLoader).
        return len(self.image_paths)

    def get_image_paths(self, path, phase):
        paths = list()
        for root, dirs, files in os.walk(path):
            for file_name in files:
                if phase in file_name:
                    paths.append(os.path.join(root, file_name))
                elif phase is "Fake":
                    if "Clay" in file_name or "Gltn" in file_name:
                        paths.append(os.path.join(root, file_name))
        return paths


if __name__=='__main__':
    train_data = ImageLoader('dataset/train', normalize_each=True)
    print(len(train_data))
    image, label = train_data[0]
    print (type(image))
