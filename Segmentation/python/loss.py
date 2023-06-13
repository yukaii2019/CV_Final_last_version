import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import copy
import cv2
from utils import mask_iou

##from torch import linalg as LA
def create_meshgrid(
        height: int,
        width: int,
        normalized_coordinates: Optional[bool] = True) -> torch.Tensor:
    """Generates a coordinate grid for an image.

    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (Optional[bool]): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    # generate coordinates
    xs: Optional[torch.Tensor] = None
    ys: Optional[torch.Tensor] = None
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width)
        ys = torch.linspace(-1, 1, height)
    else:
        xs = torch.linspace(0, width - 1, width)
        ys = torch.linspace(0, height - 1, height)
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(
        torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2

def normPts(pts):
    pts_o = copy.deepcopy(pts)
    res = pts_o.shape
    pts_o = pts_o.reshape(-1, 2)
    pts_o[:, 0] = 2*pts_o[:, 0] - 1
    pts_o[:, 1] = 2*pts_o[:, 1] - 1
    pts_o = pts_o.reshape(res)
    return pts_o

def get_seg2ptLoss(op, gtPts, temperature=1):
    # Custom function to find the center of mass to get detected pupil or iris
    # center
    # op: BXHXW - single channel corresponding to pupil or iris predictions
    B, H, W = op.shape
    wtMap = F.softmax(op.view(B, -1)*temperature, dim=1) # [B, HXW]
    #print(wtMap.shape)

    XYgrid = create_meshgrid(H, W, normalized_coordinates=True) # 1xHxWx2
    #print(XYgrid.shape)

    if str(op.device) == 'cpu':
        xloc = XYgrid[0, :, :, 0].reshape(-1)
        yloc = XYgrid[0, :, :, 1].reshape(-1)
    else:
        xloc = XYgrid[0, :, :, 0].reshape(-1).cuda()
        yloc = XYgrid[0, :, :, 1].reshape(-1).cuda()

    #print(xloc.shape)
    xpos = torch.sum(wtMap*xloc, -1, keepdim=True)
    ypos = torch.sum(wtMap*yloc, -1, keepdim=True)

    #print(xpos.shape)
    predPts = torch.stack([xpos, ypos], dim=1).squeeze()

    #print(predPts.shape)

    loss = F.l1_loss(predPts, gtPts, reduction='none')
    return loss, predPts

def CE(ip, target):
    mxLabel = ip.shape[0]
    allClasses = np.arange(mxLabel, )
    labelsPresent = np.unique(target.cpu().numpy())
    rmIdx = allClasses[~np.in1d(allClasses, labelsPresent)]
    if rmIdx.size > 0:
        loss = F.cross_entropy(ip.view(1, mxLabel, -1), target.view(1, -1), ignore_index=rmIdx.item())
    else:
        loss = F.cross_entropy(ip.view(1, mxLabel, -1), target.view(1, -1))
    loss = torch.mean(loss)
    return loss

def GDiceLoss(ip, target, norm=F.softmax):

    mxLabel = ip.shape[1]
    allClasses = np.arange(mxLabel, )
    labelsPresent = np.unique(target.cpu().numpy())

    Label = (np.arange(mxLabel) == target.cpu().numpy()[..., None]).astype(np.uint8)
    Label = np.moveaxis(Label, 3, 1)
    target = torch.from_numpy(Label).cuda().to(ip.dtype)

    loc_rm = np.where(~np.in1d(allClasses, labelsPresent))[0]

    assert ip.shape == target.shape
    ip = norm(ip, dim=1) # Softmax or Sigmoid over channels
    ip = torch.flatten(ip, start_dim=2, end_dim=-1)
    target = torch.flatten(target, start_dim=2, end_dim=-1).cuda().to(ip.dtype)
    numerator = ip*target
    denominator = ip + target

    # For classes which do not exist in target but exist in input, set weight=0
    class_weights = 1./(torch.sum(target, dim=2)**2).clamp(1e-5)
    if loc_rm.size > 0:
        for i in np.nditer(loc_rm):
            class_weights[:, i] = 0
    A = class_weights*torch.sum(numerator, dim=2)
    B = class_weights*torch.sum(denominator, dim=2)
    dice_metric = 2.*torch.sum(A, dim=1)/torch.sum(B, dim=1)
    return torch.mean(1 - dice_metric.clamp(1e-5))

def get_segLoss(op, target, cond, beta):
    # Custom function to iteratively go over each sample in a batch and
    # compute loss.
    B = op.shape[0]
    loss_seg = []
    for i in range(0, B):
        if cond[i] == 1:

            l_cE = CE(op[i, ...], target[i, ...])
            l_gD = GDiceLoss(op[i, ...].unsqueeze(0), target[i, ...].unsqueeze(0), F.softmax)

            loss_seg.append(beta * l_gD + l_cE)

    if len(loss_seg) > 0:
        return torch.sum(torch.stack(loss_seg))/torch.sum(cond.to(torch.float32))
    else:
        return torch.tensor(0.0,requires_grad=True)



# class segLoss(nn.Module):
#     def __init__(self):
#         super(segLoss, self).__init__()

#     def CE(self, ip, target):
#         mxLabel = ip.shape[0]
#         allClasses = np.arange(mxLabel, )
#         labelsPresent = np.unique(target.cpu().numpy())
#         rmIdx = allClasses[~np.in1d(allClasses, labelsPresent)]
#         if rmIdx.size > 0:
#             loss = F.cross_entropy(ip.view(1, mxLabel, -1), target.view(1, -1), ignore_index=rmIdx.item())
#         else:
#             loss = F.cross_entropy(ip.view(1, mxLabel, -1), target.view(1, -1))
#         loss = torch.mean(loss)
#         return loss

#     def GDiceLoss(self, ip, target, norm=F.softmax):
#         mxLabel = ip.shape[1]
#         allClasses = np.arange(mxLabel, )
#         labelsPresent = np.unique(target.cpu().numpy())

#         Label = (np.arange(mxLabel) == target.cpu().numpy()[..., None]).astype(np.uint8)
#         Label = np.moveaxis(Label, 3, 1)
#         target = torch.from_numpy(Label).cuda().to(ip.dtype)

#         loc_rm = np.where(~np.in1d(allClasses, labelsPresent))[0]

#         assert ip.shape == target.shape
#         ip = norm(ip, dim=1) # Softmax or Sigmoid over channels
#         ip = torch.flatten(ip, start_dim=2, end_dim=-1)
#         target = torch.flatten(target, start_dim=2, end_dim=-1).cuda().to(ip.dtype)
#         numerator = ip*target
#         denominator = ip + target

#         # For classes which do not exist in target but exist in input, set weight=0
#         class_weights = 1./(torch.sum(target, dim=2)**2).clamp(1e-5)
#         if loc_rm.size > 0:
#             for i in np.nditer(loc_rm):
#                 class_weights[:, i] = 0
#         A = class_weights*torch.sum(numerator, dim=2)
#         B = class_weights*torch.sum(denominator, dim=2)
#         dice_metric = 2.*torch.sum(A, dim=1)/torch.sum(B, dim=1)
#         return torch.mean(1 - dice_metric.clamp(1e-5))

#     def forward(self, op, target, mask, beta): 
#         B = op.shape[0]
#         loss_seg = []
#         for i in range(0, B):
#             if mask[i] == 1:

#                 l_cE = self.CE(op[i, ...], target[i, ...])
#                 l_gD = self.GDiceLoss(op[i, ...].unsqueeze(0), target[i, ...].unsqueeze(0), F.softmax)

#                 loss_seg.append(beta * l_gD + l_cE)

#         if len(loss_seg) > 0:
#             return torch.sum(torch.stack(loss_seg))/torch.sum(mask.to(torch.float32))
#         else:
#             return torch.tensor(0)

        

# def get_ptLoss(ip_vector, target_vector, cond):
#     # Custom function to iteratively find L1 distance over valid samples
#     # Note, pupil centers are assumed to be normalized between -1 and 1
#     B = ip_vector.shape[0]
#     loss_pt = []
#     for i in range(0, B):
#         if cond[i] == 1:
#             loss_pt.append(F.l1_loss(ip_vector[i, ...], target_vector[i, ...]))

#     if len(loss_pt) > 0:
#         return torch.sum(torch.stack(loss_pt))/torch.sum(cond.to(torch.float32))
#     else:
#         return torch.tensor(0)

# class ptLoss(nn.Module):
#     def __init__(self):
#         super(ptLoss, self).__init__()
#     def forward(self, input, target, mask): 
#         loss_pt = []
#         B = input.shape[0]
#         for i in range(0, B):
#             if mask[i] == 1:
#                 loss_pt.append(F.l1_loss(input[i, ...], target[i, ...]))

#         if len(loss_pt) > 0:
#             return torch.sum(torch.stack(loss_pt))/torch.sum(mask.to(torch.float32))
#         else:
#             return torch.tensor(0)


# not workable
# def get_ellmask_loss(pred_pos, target, mask):
#     """
#     ellparm.shape [B, 5]
#     target.shape [B, H,W]
#     mask.shape [B,1]
#     """
#     B = target.shape[0]
#     h, w = target.shape[1], target.shape[2]

#     loss_ellmask = []


#     pred_pos = torch.clone(pred_pos).cpu().detach().numpy()
#     target = torch.clone(target).cpu().detach().numpy()
#     pred_pos = pred_pos * np.array([w, h, h/2, w/2, 1]) # x y h w

#     for i in range(0, B):
#         if mask[i] == 1:
#             center_coordinates = (int(np.clip(pred_pos[i][0], 0, w)), int(np.clip(pred_pos[i][1], 0, h)))
#             axesLength = (int(np.clip(pred_pos[i][3], 0, w/2)), int(np.clip(pred_pos[i][2], 0, h/2)))
#             angle = np.arcsin(pred_pos[i][4]) * (180 / np.pi)
            
#             pred_ell_mask = np.zeros((h,w), dtype=np.uint8)
#             pred_ell_mask = cv2.ellipse(pred_ell_mask, center_coordinates, axesLength, angle, 0, 360, 255, -1) 
 
#             iou = mask_iou(pred_ell_mask, target[i])
#             loss_ellmask.append(iou)

#     if len(loss_ellmask) > 0:
#         return torch.sum(torch.tensor(loss_ellmask))/torch.sum(mask.to(torch.float32))
#     else:
#         return torch.tensor(0)

########################################################################################################################
# calculate 2-Wasserstein distance
# follow the paper Ellipse Detection and Localization with Applications to Knots in Sawn Lumber Images

# def get_cov_sqrt_cov(pos):
#     cos = torch.cos(torch.arcsin(pos[4]))
#     sin = pos[4]


#     R = torch.stack([torch.stack([cos, -sin]),
#                      torch.stack([sin, cos])])

#     sqrt_Sigma=torch.stack([torch.stack([pos[3]/2, torch.tensor(0).cuda()]),
#                             torch.stack([torch.tensor(0).cuda(), pos[2]/2])]) 

#     sqrt_Cov = torch.matmul(torch.matmul(R, sqrt_Sigma), R.t())

#     # Cov = torch.matmul(sqrt_Cov, sqrt_Cov)

#     return sqrt_Cov

# def get_gaussian_IoU_ellipse(pos, target_pos, mask):
#     """
#     pos.shape = [B,5]
#     (x, y, h, w, theta)
#     """
#     B = pos.shape[0]
#     loss = []
#     for i in range(0,B):
#         if mask[i] == 1:
#             sqrt_Cov = get_cov_sqrt_cov(pos[i])
#             sqrt_Cov_tar = get_cov_sqrt_cov(target_pos[i])
            
#             trace_term = LA.matrix_norm(sqrt_Cov - sqrt_Cov_tar, ord='fro')
            
#             trace_term = trace_term * trace_term

#             mu_pos = pos[i][:2]
#             mu_tar = target_pos[i][:2]
#             diff = mu_pos - mu_tar
#             mean_term = torch.sum(torch.mul(diff, diff))  # scalar
#             loss.append(mean_term + trace_term)

    
#     if len(loss) > 0:
#         return torch.sum(torch.stack(loss))/torch.sum(mask.to(torch.float32))
#     else:
#         return torch.tensor(0) 
########################################################################################################################

# the function used to check back propagation
# def getBack(var_grad_fn):
#     print(var_grad_fn)
#     for n in var_grad_fn.next_functions:
#         if n[0]:
#             try:
#                 tensor = getattr(n[0], 'variable')
#                 print(n[0])
#                 print('Tensor with grad found:', tensor)
#                 print(' - gradient:', tensor.grad)
#                 print()
#             except AttributeError as e:
#                 getBack(n[0])


if __name__ == '__main__':

    device = torch.device('cuda')

    B = 3
    H = 480
    W = 640

    # op = torch.rand(B, 2, H, W,requires_grad=True).to(device)
    # pupil_center = torch.rand(B, 2).to(device)

    # l_seg2pt_pup, pred_c_seg_pup = get_seg2ptLoss(op[:, 1, ...], normPts(pupil_center), temperature=4)
    # print(l_seg2pt_pup.shape)
    # print(pred_c_seg_pup.shape)

    # loc_onlyMask = torch.randint(0,2,(B,1)).to(device)
    # loc_onlyMask = torch.zeros((B,1)).to(device)
    # print(loc_onlyMask)
    # elOut = torch.rand(B,10, requires_grad = True).to(device)

    # l_pt = get_ptLoss(elOut[:, 5:7], normPts(pupil_center), 1-loc_onlyMask)
    # l_pt.backward()
    # print(l_pt)


    # #l_ellipse = get_ptLoss(elOut, elNorm.view(-1, 10), loc_onlyMask)

    # cri_ptLoss = ptLoss()
    # l_pt = cri_ptLoss(elOut[:, 5:7], normPts(pupil_center), 1-loc_onlyMask) 
    # print(l_pt)

    # beta = 0.4
    # target = torch.randint(0, 2, (B, H, W)).to(device)
    # l_seg = get_segLoss(op, target, loc_onlyMask, beta)
    # l_seg.backward()
    # getBack(l_seg.grad_fn) 

    # cri_segLoss = segLoss()
    # l_seg = cri_segLoss(op, target, loc_onlyMask, beta)
    # print(l_seg) 

    # a = torch.tensor(0)
    # print(a)
    # print(torch.tensor(0).item())


    # l_ellmask = get_ellmask_loss(elOut[:,:5], target, loc_onlyMask)
    # l_ellmask.backward()
    # print(l_ellmask)

    # pos = torch.tensor([[2, 3, 6, 8, 0.5],[5, 2, 3, 4, 0.5]], dtype = torch.float32, requires_grad=True).to(device)
    # pos_tar = torch.tensor([[2, 3, 6, 7, 0.5],[5, 2, 3, 4, 0.5]]).to(device)
    # mask = torch.tensor([1, 0]).to(device)
    # # print(pos)
    # # print(pos_tar)
    
    # giouLoss = get_gaussian_IoU_ellipse(pos, pos_tar, mask) 
    # print(giouLoss)
    # giouLoss.backward()
    # print(pos.grad_fn)
    # #getBack(giouLoss.grad_fn)

    # #print(giouLoss)
