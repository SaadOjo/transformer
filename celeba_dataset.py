import random
from PIL import Image
from torch.utils.data import Dataset
import torch

class CelebADataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None, seed=None, type='train'):
        """
        Args:
            img_dir (str): Path to the directory containing images.
            label_file (str): Path to the text file containing image filenames and their corresponding identities.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.type = type
        
        # Read the label file
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # Reduce the dataset size by getting random subset of 20000 samples
        if not seed:
            seed = 42
        random.seed(seed)
        random.shuffle(lines)

        if type == 'train': # get the first 20000 samples
            lines = lines[:20000]
        elif type == 'test' or 'embedding_gen': # get the last 20000 samples
            lines = lines[-20000:]
        self.labels_with_image_names = {}
        for line in lines:
            parts = line.strip().split()
            img = parts[0]
            label = int(parts[1])
            if label not in self.labels_with_image_names:
                self.labels_with_image_names[label] = []
            self.labels_with_image_names[label].append(img) # store the image names for each label
        
        # filter out labels with less than 2 samples
        self.labels_with_image_names = {label: image_names for label, image_names in self.labels_with_image_names.items() if len(image_names) >= 2}

        # Populate self.data
        self.data = []
        for label, image_names in self.labels_with_image_names.items():
            for image in image_names:
                self.data.append({"image": image, "label": label})

        # Create a dictionary mapping labels to indices in self.data
        self.label_to_indices = {}
        for i, sample in enumerate(self.data):
            label = sample["label"]
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(i)

        del self.labels_with_image_names  

    def __len__(self):
        """Return the total number of samples."""
        if self.type == 'embedding_gen':
            return len(self.label_to_indices) # return number of unique identities
        else: 
            return len(self.data)
    
    def __getitem__(self, idx):
        """
        Fetch a triplet (anchor, positive, negative).
        
        Args:
            idx (int): Index of the anchor sample.
            
        Returns:
            anchor, positive, negative (Tensors): Transformed images.
        """
        if self.type == 'embedding_gen':
            label = list(self.label_to_indices.keys())[idx]
            images_list = []
            for img_idx in self.label_to_indices[label]:
                image = Image.open(f"{self.img_dir}/{self.data[img_idx]['image']}").convert('RGB')
                image = self.transform(image)
                images_list.append(image)
                images_tensor = torch.stack(images_list, dim=0)
            return images_tensor, label
        
        if self.type == 'train':
            image = Image.open(f"{self.img_dir}/{self.data[idx]['image']}").convert('RGB')
            image = self.transform(image)
            label = self.data[idx]['label']
            return image, label   
    
        anchor_data = self.data[idx]
        anchor_label = anchor_data["label"]
    
        # Select a positive sample

        positive_idx = random.choice(self.label_to_indices[anchor_label])
        while positive_idx == idx:  # Ensure positive is not the same as anchor
            positive_idx = random.choice(self.label_to_indices[anchor_label])
        
        # Select a negative sample
        negative_label = random.choice(list(self.label_to_indices.keys()))
        while negative_label == anchor_label:
            negative_label = random.choice(list(self.label_to_indices.keys()))
        negative_idx = random.choice(self.label_to_indices[negative_label])
        
        # Load images
        anchor = Image.open(f"{self.img_dir}/{anchor_data['image']}").convert('RGB')
        positive = Image.open(f"{self.img_dir}/{self.data[positive_idx]['image']}").convert('RGB')
        negative = Image.open(f"{self.img_dir}/{self.data[negative_idx]['image']}").convert('RGB')
        
        # Apply transformations
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative


if __name__ == "__main__":
    
    dataset = CelebADataset(img_dir='./data/celeba/img_align_celeba', label_file='./data/celeba/identity_CelebA.txt')