import os
import random
from torch.utils.data import Dataset
from PIL import Image
import timm
import xml.etree.ElementTree as ET

random.seed(0)

class Imagenet(Dataset):
    def __init__(self, root_dir, transform=None, full=False):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_to_index = {} 

        label_names = sorted(os.listdir(root_dir))
        random_labels = random.sample(label_names, 100) if not full else label_names
        self.label_to_index = {label: idx for idx, label in enumerate(random_labels)}

        for label in random_labels:
            img_folder = os.path.join(root_dir, label)
            if os.path.isdir(img_folder):
                for img_file in os.listdir(img_folder):
                    if img_file.endswith('.JPEG'):
                        self.images.append(os.path.join(img_folder, img_file))
                        self.labels.append(self.label_to_index[label]) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB') 
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class ValidationDataset(Dataset):
    def __init__(self, root_dir, annotations_dir, label_to_index, transform=None):
        self.root_dir = root_dir
        self.annotations_dir = annotations_dir
        self.label_to_index = label_to_index
        self.transform = transform
        self.images = []
        self.labels = []

        for annotation_file in os.listdir(annotations_dir):
            if annotation_file.endswith('.xml'):
                annotation_path = os.path.join(annotations_dir, annotation_file)

                tree = ET.parse(annotation_path)
                root = tree.getroot()

                filename = root.find('filename').text + '.JPEG'
                img_path = os.path.join(root_dir, filename)

                object_name = root.find('object/name').text
                if object_name in label_to_index:
                    self.images.append(img_path)
                    self.labels.append(label_to_index[object_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB') 
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transform(train=True):
    return timm.data.create_transform(
        input_size=(3, 224, 224),
        is_training=train,
        auto_augment='rand-m9-mstd0.5-inc1', 
        re_prob=0.25,  
    )