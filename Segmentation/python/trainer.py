import os
import csv
import random
import glob

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tqdm


from module import *
from data import get_dataloader
from logger import logger
from utils import *

class Trainer:
    def __init__(self, args):
        '''
            args:
            ## input file
                dataset_dir="/home/ykhsieh/CV/final/dataset/"
                label_data="/home/ykhsieh/CV/final/dataset/data.json"

            ## output file
                val_imgs_dir="./log-${time}/val_imgs"
                learning_curv_dir="./log-${time}/val_imgs"
                check_point_root="./log-${time}/checkpoints"
                log_root="./log-${time}"

            ## others
                batch_size=2
                lr=0.0001
                num_epochs=150
                milestones = [50, 100, 150]
        '''

        set_seed(9527)

        self.args = args
        self.device = torch.device('cuda')

    ########################################################################
    #                   Use your model                                     #
    ########################################################################  
        self.model = DenseNet2D().to(self.device)
        #self.model = mobilenet().to(self.device)
        #self.model = ResNet50().to(self.device)
    ########################################################################
    #                   Use your model                                     #
    ########################################################################  

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        self.train_loader = get_dataloader(args.dataset_dir, args.label_data, batch_size=args.batch_size, split='train')
        self.val_loader   = get_dataloader(args.dataset_dir, args.label_data, batch_size=args.batch_size, split='val')        


        self.criterion2 = nn.CrossEntropyLoss()

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestones, gamma=0.1)
        
        self.train_loss_list = {'total': [], 'iou' : []}
        self.val_loss_list = {'total': [], 'iou' : []}

    def plot_learning_curve(self, result_list, name='train'):
        for (type_, list_) in result_list.items():
            plt.plot(range(len(list_)), list_, label=f'{name}_{type_}_value')
            plt.title(f'{name} {type_}')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend(loc='best')
            plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.savefig(os.path.join(self.args.learning_curv_dir , f'{name}_{type_}.png'))
            plt.close()


    def save_checkpoint(self):
        for pth in glob.glob(os.path.join(self.args.check_point_root, '*.pth')):
            os.remove(pth)
        logger.info(f'[{self.epoch + 1}/{self.args.num_epochs}] Save best model to {self.args.check_point_root} ...')
        torch.save(self.model.state_dict(), os.path.join(self.args.check_point_root, f'model_best_{int(self.best_score*10000)}.pth')) 



    def visualize(self, pred_mask, images, name = 'train'):
        pred_mask = pred_mask.max(1, keepdim=False)[1].cpu().numpy()[0]
        pred_mask = (255*pred_mask).astype(np.uint8)
        cv2.imwrite(os.path.join(self.args.val_imgs_dir, f'{name}_mask{self.epoch}.jpg'), pred_mask)

        image = images.cpu().detach().numpy()[0][0]
        image = (255*image).astype(np.uint8)
        cv2.imwrite(os.path.join(self.args.val_imgs_dir, f'{name}_image{self.epoch}.jpg'), image)


    def train_epoch(self):

        total_loss = 0.0
        num_of_open_eye = 0
        self.iou = 0.0

        self.model.train()
        for batch, data in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=80, leave=False):

            images, mask, conf = data['images'].to(self.device), data['mask'].to(self.device), data['conf'].to(self.device)

            openeye = torch.sum(conf == 1).detach().cpu().item()

            if openeye != 0:
                # pred_mask = self.model(images[conf == 1])
                pred_mask = self.model(images)

                loss = self.criterion2(pred_mask, mask)
                # loss = self.criterion2(pred_mask, mask[conf == 1])

                num_of_open_eye += openeye

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                self.iou += self.benchmark(pred_mask, mask[conf == 1]) 

        total_loss  /= len(self.train_loader)
        self.iou /= num_of_open_eye

        logger.info(f'[{self.epoch + 1}/{self.args.num_epochs}] Train Loss: {total_loss:.5f} | Train iou: {self.iou:.5f}')

        self.train_loss_list['total'].append(total_loss)
        self.train_loss_list['iou'].append(self.iou)
        self.visualize(pred_mask, images, name='train')

    def benchmark(self, pred_mask,  mask):
        pred_mask = torch.clone(pred_mask.max(1, keepdim=False)[1]).cpu().detach().numpy()
        mask = torch.clone(mask).cpu().detach().numpy() 

        total_iou = 0.0
        for i in range(len(mask)):
            total_iou += mask_iou(pred_mask[i], mask[i])
        return total_iou


    def val_epoch(self):
        self.model.eval()
        
        total_loss = 0.0
        num_of_open_eye = 0
        self.iou = 0.0

        with torch.no_grad():

            for batch, data in tqdm.tqdm(enumerate(self.val_loader), total = len(self.val_loader), ncols=80, leave=False):


                images, mask,  conf = data['images'].to(self.device), data['mask'].to(self.device),  data['conf'].to(self.device)

                openeye = torch.sum(conf == 1).detach().cpu().item()
                
                if openeye != 0:
                    pred_mask = self.model(images[conf == 1])

                    loss = self.criterion2(pred_mask, mask[conf == 1])

                    num_of_open_eye += openeye

                    total_loss += loss.item()
                    self.iou += self.benchmark(pred_mask, mask[conf == 1]) 


        total_loss /=   len(self.val_loader)
        self.iou /= num_of_open_eye

        
        logger.info(f'[{self.epoch + 1}/{self.args.num_epochs}] Val Loss: {total_loss:.5f} | Val iou: {self.iou:.5f}')


        self.val_loss_list['total'].append(total_loss)
        self.val_loss_list['iou'].append(self.iou)

        self.visualize(pred_mask, images, name='val')

        

    def train(self):
        self.epoch = 0
        self.best_score = None

        for self.epoch in range(self.args.num_epochs):
            self.alpha = linVal(self.epoch, (0, self.args.num_epochs), (0, 1), 0)
            self.beta = 1-self.alpha

            self.train_epoch()
            self.plot_learning_curve(self.train_loss_list, name='train')

            self.val_epoch()
            self.plot_learning_curve(self.val_loss_list, name='val')

            self.scheduler.step()

            if self.best_score == None or self.iou > self.best_score:
                self.best_score = self.iou
                self.save_checkpoint()
                



    
            


    

