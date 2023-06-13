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
                    
                    # for name in added_names:
                    #     mask = Image.open(os.path.join(self.dataset_dir , name + ".png")).convert('L')
                    #     one += np.sum(mask!=0)

                except:
                    print(f'Labels are not available for {image_folder}')
        
        # print(f'Number of {self.split} images is {self.len}')
        # print(f'Number of one is {one} {one/self.len/480/640}')
        # print(f'Number of zero is {self.len*480*640-one} {(self.len*480*640-one)/self.len/480/640}')
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):

        fn_img = os.path.join(self.dataset_dir , self.image_names[index]+".jpg")
        fn_msk = os.path.join(self.dataset_dir , self.image_names[index]+".png")
        img = Image.open(fn_img).convert('L')
        mask = Image.open(fn_msk).convert('L')
        
        mask = (np.array(mask) != 0).astype(int)
    
        img = transforms.functional.pil_to_tensor(img)
        mask = transforms.ToTensor()(mask)           

        conf = self.label_data[os.path.join(self.image_names[index]+".jpg")]

        ########################################################################
        #                   data augmentation start                            #
        ########################################################################

        #if self.split == 'train' or self.split == 'train_val':
#
        #    # random gaussian blur
        #    if random.random() < 0.5:
        #        img = transforms.GaussianBlur(3, sigma=(2, 7))(img)
#
        #    # random gamma correction 
        #    if random.random() < 0.5:
        #        img = transforms.functional.adjust_gamma(img, random.choice([0.6, 0.8, 1.2, 1.4]))
#
        #    # random rotate
        #    #r = transforms.RandomRotation.get_params((-20, 20))
        #    #img = TF.rotate(img, r)
        #    #mask = TF.rotate(mask, r)
#
        #    # # random crop
        #    # i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(192, 256))
        #    # img = TF.crop(img, i, j, h, w)
        #    # mask = TF.crop(mask, i, j, h, w)
#
        #    # random horizontal flip
        #    #if random.random() < 0.5:
        #    #    img = TF.hflip(img)
        #    #    mask = TF.hflip(mask)
        #
        ########################################################################
        #                   data augmentation end                             #
        ########################################################################

        ########################################################################
        #                   preprocessing start                                #
        ########################################################################
        # img = TF.gaussian_blur(img, 13)
        # img = TF.adjust_gamma(img, 0.4)
        # img = TF.gaussian_blur(img, 13)
        ########################################################################
        #                   preprocessing end                                  #
        ########################################################################

        mask = torch.squeeze(mask, 0)
        img = transforms.Resize((192, 256))(img)
        img = (img / 255).to(torch.float32)
        mask = mask.to(int)
        conf = int(conf)
        return {
            'images': img,
            'mask': mask,
            'conf': conf,
        }


def imshow(inp, fn=None, mul255 = False):
    inp = inp.numpy().transpose((1, 2, 0))
    if mul255: inp = inp*255
    im = Image.fromarray((inp).astype(np.uint8))
    im.save(fn)

if __name__ == '__main__':
    val_loader = get_dataloader('/home/ystseng/course/CV/final/Dataset', '/home/ystseng/course/CV/final/h_2/CV_Final_segmentation/conf.json', batch_size=4, split='val')
    train_loader = get_dataloader('/home/ystseng/course/CV/final/Dataset', '/home/ystseng/course/CV/final/h_2/CV_Final_segmentation/conf.json', batch_size=4, split='train')
    train_val_loader = get_dataloader('/home/ystseng/course/CV/final/Dataset', '/home/ystseng/course/CV/final/h_2/CV_Final_segmentation/conf.json', batch_size=4, split='train_val')
    
    data = iter(train_loader).next()

    print(data['images'].shape, data['mask'].shape, data['conf'].shape)

    images = data['images']
    mask = data['mask'] 
    conf = data['conf']

    images = images.cpu().numpy()
    images = (255*images).astype(np.uint8)

    mask = mask.unsqueeze(1)
    mask = mask.cpu().numpy()
    mask = (255*mask).astype(np.uint8)

    h, w = images.shape[2], images.shape[3]

    imshow(torchvision.utils.make_grid(torch.tensor(images)), fn="train_set.png", mul255 = False)
    imshow(torchvision.utils.make_grid(torch.tensor(mask)), fn="train_set_mask.png", mul255 = False)
