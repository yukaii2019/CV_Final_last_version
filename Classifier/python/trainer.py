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
from sklearn import metrics
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

        self.model = efficientnet().to(self.device)
        #self.model.load_state_dict(torch.load("/home/ykhsieh/CV/final/classifier/log-20230603153001/checkpoints/model_best_9997.pth", map_location=self.device))

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        self.train_loader = get_dataloader(args.dataset_dir, args.label_data, batch_size=args.batch_size, split='train')
        self.val_loader   = get_dataloader(args.dataset_dir, args.label_data, batch_size=args.batch_size, split='val')        

        self.criterion1 = nn.MSELoss() 
        self.criterion2 = nn.CrossEntropyLoss(weight = torch.tensor([0.9, 0.1]).to(self.device))
        #self.criterion2 = nn.CrossEntropyLoss()
        #self.criterion2 = nn.CrossEntropyLoss(weight = torch.tensor([0.1, 0.9]).to(self.device))

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestones, gamma=0.1)
        
        self.train_loss_list = {'total': [], 'acc':[], 'atnr':[], 'auc':[]}
        self.val_loss_list = {'total': [], 'acc': [], 'atnr': [], 'auc':[]}

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

    def train_epoch(self):

        total_loss = 0.0
        correct = 0 

        label_validity = []
        output_conf = []

        self.model.train()
        for batch, data in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=80, leave=False):

            images, conf = data['images'].to(self.device), data['conf'].to(self.device)
            
            pred_conf = self.model(images)

            loss = self.criterion2(pred_conf, conf)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            

            pred = pred_conf.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(conf.view_as(pred)).sum().item()

            total_loss += loss.item()

            label_validity.extend(conf.clone().detach().cpu().numpy())
            #output_conf.extend(torch.clone(pred_conf.max(1, keepdim=False)[1]).cpu().detach().numpy()) 
            output_conf.extend(torch.clone(pred_conf).softmax(1)[:,1].cpu().detach().numpy()) 
            

        total_loss /= len(self.train_loader)
        self.acc = 100. * correct / len(self.train_loader.dataset)

        tn_rates = true_negative_curve(np.array(output_conf), np.array(label_validity))
        self.atnr = np.mean(tn_rates)

        fpr, tpr, thresholds = metrics.roc_curve(np.array(label_validity), np.array(output_conf), pos_label=1)
        self.auc = metrics.auc(fpr, tpr)

        # Calculate the G-mean
        gmean = np.sqrt(tpr * (1 - fpr))

        # Find the optimal threshold
        index = np.argmax(gmean)
        thresholdOpt = thresholds[index]


        self.train_loss_list['total'].append(total_loss)
        self.train_loss_list['acc'].append(self.acc)
        self.train_loss_list['atnr'].append(self.atnr)
        self.train_loss_list['auc'].append(self.auc)

        logger.info(f'[{self.epoch + 1}/{self.args.num_epochs}] Train Loss: {total_loss:.5f} | Train Acc: {self.acc:.5f} | Train atnr: {self.atnr:.5f} | Train auc: {self.auc:.5f} | thresholdOpt: {thresholdOpt:.5f}')
        
    
    def val_epoch(self):
        self.model.eval()
        
        total_loss = 0.0
        correct = 0

        label_validity = []
        output_conf = []

        with torch.no_grad():

            for batch, data in tqdm.tqdm(enumerate(self.val_loader), total = len(self.val_loader), ncols=80, leave=False):


                images,  conf = data['images'].to(self.device), data['conf'].to(self.device)

                pred_conf = self.model(images)

                loss = self.criterion2(pred_conf, conf)

                pred = pred_conf.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(conf.view_as(pred)).sum().item()

                total_loss += loss.item()

                label_validity.extend(conf.clone().detach().cpu().numpy())
                #output_conf.extend(torch.clone(pred_conf.max(1, keepdim=False)[1]).cpu().detach().numpy()) 
                output_conf.extend(torch.clone(pred_conf).softmax(1)[:,1].cpu().detach().numpy()) 


        total_loss /=  len(self.val_loader)
        self.acc = 100. * correct / len(self.val_loader.dataset)
        
        tn_rates = true_negative_curve(np.array(output_conf), np.array(label_validity))
        self.atnr = np.mean(tn_rates)

        fpr, tpr, thresholds = metrics.roc_curve(np.array(label_validity), np.array(output_conf), pos_label=1)
        self.auc = metrics.auc(fpr, tpr)
        
        self.val_loss_list['total'].append(total_loss)
        self.val_loss_list['acc'].append(self.acc)
        self.val_loss_list['atnr'].append(self.atnr)
        self.val_loss_list['auc'].append(self.auc)

        logger.info(f'[{self.epoch + 1}/{self.args.num_epochs}] val Loss: {total_loss:.5f} | val Acc: {self.acc:.5f} | val atnr: {self.atnr:.5f} | val auc: {self.auc:.5f}') 
        

    def train(self):
        self.epoch = 0
        self.best_score = None

        for self.epoch in range(self.args.num_epochs):
            self.train_epoch()
            self.plot_learning_curve(self.train_loss_list, name='train')

            self.val_epoch()
            self.plot_learning_curve(self.val_loss_list, name='val')

            self.scheduler.step()

            if self.best_score == None or self.auc > self.best_score:
                self.best_score = self.auc 
                self.save_checkpoint()
                



    
            


    

