import os
import glob
import torch
import numpy as np
import cv2
from tqdm import tqdm
import torchvision.transforms as transforms
from argparse import ArgumentParser
from PIL import Image
from module import DenseNet2D

def main(args):


    device = torch.device('cuda')

    ########################################################################
    #                   Use your model                                     #
    ########################################################################  
    model_seg = DenseNet2D()
    ########################################################################
    #                   Use your model                                     #
    ########################################################################  
    
    model_seg.load_state_dict(torch.load(args.segmatation_checkpoint, map_location=torch.device('cpu')))
    model_seg = model_seg.to(device)
    model_seg.eval()

    subjects = ['S5', 'S6', 'S7', 'S8']
    with torch.no_grad():
        sequence_idx = 0
        for subject in subjects:
            for action_number in range(0, 26):
                image_folder = os.path.join(args.dataset_path, subject, f'{action_number + 1:02d}')
                sequence_idx += 1
                nr_image = 0
                try:
                    nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
                except:
                    continue

                directory_path = os.path.join(args.output_path , subject, f'{action_number + 1:02d}')
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                
                conf = []

                for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {image_folder}'):
                    fn_img = os.path.join(image_folder, f'{idx}.jpg')
                    img = Image.open(fn_img).convert('L')
                    img = transforms.functional.pil_to_tensor(img)
        ########################################################################
        #                   preprocessing start                                #
        ########################################################################  
                    # img = transforms.functional.gaussian_blur(img, 13)
                    # img = transforms.functional.adjust_gamma(img, 0.4)
                    # img = transforms.functional.gaussian_blur(img, 13)
        ########################################################################
        #                   preprocessing end                                  #
        ########################################################################  

                    img = transforms.Resize((192, 256))(img)
                    img = (img / 255).to(torch.float32).to(device).unsqueeze(0)

                    
                    pred_mask = model_seg(img)


                    pred_mask = pred_mask.max(1, keepdim=False)[1].cpu().detach().numpy()
                    
                    pred_mask = (255*pred_mask).astype(np.uint8).transpose(1,2,0)
                    
                    cv2.imwrite(os.path.join(args.output_path, subject, f'{action_number + 1:02d}', f'{idx}.png'), pred_mask)

                    conf.append(int(1))
                
                txt_out_path = os.path.join(args.output_path, subject, f'{action_number + 1:02d}', 'conf.txt')
                with open(txt_out_path, 'w') as fp:
                    for item in conf:
                        fp.write("%s\n" % item)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--segmatation_checkpoint", type=str)
    args = parser.parse_args()
    main(args)