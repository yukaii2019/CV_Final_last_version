

import os
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms.functional as TF
import random
import torchvision

def get_dataloader(dataset_dir, label_data, batch_size=1, split='test'):
    dataset = mydataset(dataset_dir, label_data, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train' or split=='val' or split=='train_val'), num_workers=12, pin_memory=True)
    return dataloader


class mydataset(Dataset):
    def __init__(self, dataset_dir, label_data,  split='test'):
        super(mydataset).__init__()
        
        self.dataset_dir = dataset_dir
        self.split = split

        self.image_names = []
        self.len = 0

        self.seqs = [x for x in range(26)]
        self.subjects = ['S1', 'S2', 'S3', 'S4']

        with open(label_data) as file:
            self.label_data = json.load(file)

        one = 0
        for subject in self.subjects :
            for seq in self.seqs:
                image_folder = os.path.join(subject, f'{seq + 1:02d}')
                try:
                    names = [os.path.splitext(os.path.join(image_folder, name))[0] for name in os.listdir(os.path.join(self.dataset_dir, image_folder)) if name.endswith('.jpg')]
                    mid = int(len(names) * 0.8)
                    
                    if split == 'train':
                        added_names = names[:mid] 
                    elif split == 'val':
                        added_names = names[mid:] 
                    elif split == 'train_val':
                        added_names = names
                    
                    self.image_names.extend(added_names)
                    self.len += len(added_names)    
                    
                    for name in added_names:
                        one += self.label_data[name+".jpg"]

                except:
                    print(f'Labels are not available for {image_folder}')
        
    
        print(f'Number of {self.split} images is {self.len}')
        print(f'Number of one is {one} {one/self.len}')
        print(f'Number of zero is {self.len-one} {(self.len-one)/self.len}')

    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        fn_img = os.path.join(self.dataset_dir , self.image_names[index]+".jpg")
        img = Image.open(fn_img).convert('L')
        conf = self.label_data[os.path.join(self.image_names[index]+".jpg")]
        
        img = TF.pil_to_tensor(img)

        #img = TF.equalize(img)
        img = TF.gaussian_blur(img, 13)
        img = TF.adjust_gamma(img, 0.4)
        img = TF.gaussian_blur(img, 13)
        

        if self.split == 'train' or self.split == 'train_val':
            # # random gaussian blur
            # if random.random() < 0.5:
            #     img = transforms.GaussianBlur(3, sigma=(2, 7))(img)
            # # random gamma correction 
            # if random.random() < 0.5:
            #     img = transforms.functional.adjust_gamma(img, random.choice([0.6, 0.8, 1.2, 1.4]))
            
            # random rotate
            r = transforms.RandomRotation.get_params((-20, 20))
            img = TF.rotate(img, r)
            
            # random horizontal flip
            if random.random() < 0.5:
                img = TF.hflip(img)

        img = transforms.Resize((192, 256))(img)
        img = (img/255).to(torch.float32)
        conf = int(conf)
        return {
            'images': img,
            'conf': conf,
        }

def imshow(inp, fn=None, mul255 = False):
    inp = inp.numpy().transpose((1, 2, 0))
    if mul255: inp = inp*255
    im = Image.fromarray((inp).astype(np.uint8))
    im.save(fn)

if __name__ == '__main__':
    train_loader = get_dataloader('/home/ykhsieh/CV/final/dataset', '/home/ykhsieh/CV/final/dataset/conf.json', batch_size=4, split='train')
    val_loader = get_dataloader('/home/ykhsieh/CV/final/dataset', '/home/ykhsieh/CV/final/dataset/conf.json', batch_size=4, split='val')
    train_val_loader = get_dataloader('/home/ykhsieh/CV/final/dataset', '/home/ykhsieh/CV/final/dataset/conf.json', batch_size=4, split='train_val')


    print(train_loader.dataset.len)
    data = iter(train_loader).next()


    print(data['images'].shape, data['conf'].shape)

    images = data['images']
    conf = data['conf']

    print(conf)
    images = images.cpu().numpy()
    images = (255*images).astype(np.uint8)


    imshow(torchvision.utils.make_grid(torch.tensor(images)), fn="train_set.png", mul255 = False)
