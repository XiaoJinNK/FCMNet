
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from dataset_loader import MyData, MyTestData
from functions import imsave
import argparse
from train import Trainer
from model.model_depth import DepthNet
from model.model_baseline import BaselineNet
from model.model_fusion import FusionNet
import time
import torchvision
from tqdm import tqdm
from torchsummary import summary
import os
from testdata import test_dataset
from saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm

configurations = {
    1: dict(
        max_iteration=600000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        spshot=20000,
        nclass=2,
        sshow=10,
    )
}

parser=argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='train', help='train or test')
parser.add_argument('--param', type=str, default=False, help='path to pre-trained parameters')
parser.add_argument('--train_dataroot', type=str, default='/media/ubuntu/新加卷/Zhen He/RGB-D/rgb-d code/dataset/SOD-RGBD/train_data-augment', help=
                                                          'path to train data')
parser.add_argument('--test_dataroot', type=str, default='/media/ubuntu/新加卷/Zhen He/RGB-D/rgb-d code/dataset/SOD-RGBD/val/DUT-RGBD', help=
                                                         'path to test data')
parser.add_argument('--snapshot_root', type=str, default='./checkpoint', help='path to snapshot')
parser.add_argument('--salmap_root', type=str, default='./sal_map/DUT-RGBD/', help='path to saliency map')
parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys())
args = parser.parse_args()
cfg = configurations
cuda = torch.cuda.is_available

"""""""""""dataset loader"""""""""
train_dataRoot = args.train_dataroot
test_dataRoot = args.test_dataroot

if not os.path.exists(args.snapshot_root):
    os.mkdir(args.snapshot_root)
if not os.path.exists(args.salmap_root):
    os.mkdir(args.salmap_root)
if args.phase == 'train':
    SnapRoot = args.snapshot_root           # checkpoint
    train_loader = torch.utils.data.DataLoader(MyData(train_dataRoot, transform=True),
                                               batch_size=4, shuffle=True, num_workers=4, pin_memory=True,drop_last=True)
else:
    MapRoot = args.salmap_root
    test_loader = torch.utils.data.DataLoader(MyTestData(test_dataRoot, transform=True),
                                   batch_size=1, shuffle=True, num_workers=4, pin_memory=True,drop_last=True)
print ('data already')
"""""""""""train_data/test_data through nets"""""""""
start_epoch = 0
start_iteration = 0
model_depth = DepthNet()
model_baseline = BaselineNet()
model_fusion = FusionNet()
# print(model_rgb)
if args.param is True:
    # ckpt = str(ckpt)
    ckpt = '60'
    model_depth.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'depth_snapshot_iter_' + ckpt + '0000.pth')))
    model_baseline.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'baseline_snapshot_iter_'+ckpt+'0000.pth')))
    model_fusion.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'ladder_snapshot_iter_'+ckpt+'0000.pth')))
else:
    # model_depth.init_weights()
    vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
    model_depth.copy_params_from_vgg16_bn(vgg16_bn)
    model_baseline.copy_params_from_vgg16_bn(vgg16_bn)
    model_fusion.init_weights()
if cuda:
   model_depth = model_depth.cuda()
   model_baseline = model_baseline.cuda()
   model_fusion = model_fusion.cuda()

if args.phase == 'train':
    optimizer_depth = optim.SGD(model_depth.parameters(), lr=cfg[1]['lr'], momentum=cfg[1]['momentum'], weight_decay=cfg[1]['weight_decay'])
    optimizer_baseline = optim.SGD(model_baseline.parameters(), lr=cfg[1]['lr'], momentum=cfg[1]['momentum'], weight_decay=cfg[1]['weight_decay'])
    optimizer_ladder = optim.SGD(model_fusion.parameters(), lr=cfg[1]['lr'], momentum=cfg[1]['momentum'], weight_decay=cfg[1]['weight_decay'])

    training = Trainer(
        cuda=cuda,
        model_depth=model_depth,
        model_baseline=model_baseline,
        model_fusion=model_fusion,
        optimizer_depth=optimizer_depth,
        optimizer_baseline=optimizer_baseline,
        optimizer_ladder=optimizer_ladder,
        train_loader=train_loader,
        max_iter=cfg[1]['max_iteration'],
        snapshot=cfg[1]['spshot'],
        outpath=args.snapshot_root,
        sshow=cfg[1]['sshow']
    )
    training.epoch = start_epoch
    training.iteration = start_iteration
    training.train()
else:
    res = []
    for id, (data, depth, img_name, img_size) in enumerate(test_loader):
        # print('testing bach %d' % id)
        inputs = Variable(data).cuda()
        depth = Variable(depth).cuda()
        n, c, h, w = inputs.size()
        # depth = torch.unsqueeze(depth, 1)
        depth = depth.view(n, 1, h, w).repeat(1, c, 1, 1)
        # depth = depth.view(n, 1, h, w)
        torch.cuda.synchronize()
        start = time.time()
        model_fusion.eval()

        h2,h3, h4, h5 = model_baseline(inputs)
        d2,d3, d4, d5 = model_depth(depth)
        p2, p3, p4, p5, p  = model_fusion(inputs, h2, h3, h4, h5, d2,d3, d4, d5)
        torch.cuda.synchronize()
        end = time.time()
        res.append(end - start)

        pred = torch.sigmoid(p)
        outputs = pred[0, 0].detach().cpu().numpy()
        imsave(os.path.join(MapRoot,img_name[0] + '.png'), outputs, img_size)
        # imsave(os.path.join(MapRoot,img_name[0][-1] + '.png'), outputs, img_size)
    time_sum = 0
    for i in res:
        time_sum += i
    print("FPS: %f" % (1.0 / (time_sum / len(res))))

    # -------------------------- validation --------------------------- #
    sal_root = MapRoot
    gt_root = test_dataRoot + '/test_masks/'
    dataset = test_dataRoot.split('/')[-1]
    test_loader = test_dataset(sal_root, gt_root)
    mae, fm, sm, em, wfm = cal_mae(), cal_fm(test_loader.size), cal_sm(), cal_em(), cal_wfm()
    for i in tqdm(range(test_loader.size)):
        sal, gt = test_loader.load_data()
        if sal.size != gt.size:
            x, y = gt.size
            sal = sal.resize((x, y))
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        gt[gt > 0.5] = 1
        gt[gt != 1] = 0
        res = sal
        res = np.array(res)
        if res.max() == res.min():
            res = res / 255
        else:
            res = (res - res.min()) / (res.max() - res.min())
        mae.update(res, gt)
        sm.update(res, gt)
        fm.update(res, gt)
        em.update(res, gt)
        wfm.update(res, gt)

    MAE = mae.show()
    maxf, meanf, _, _ = fm.show()
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    print(
        'dataset: {} MAE: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f}'.format(dataset, MAE, maxf,
                                                                                                     meanf, wfm, sm,em))
# summary(model_baseline,input_size=(3,256,256))
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# summary(model_depth,input_size=(3,256,256))
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# summary(model_fusion,input_size=[(3,256,256),(128,64,64),(256,32,32),(512,16,16),(512,8,8),(128,64,64),(256,32,32),(512,16,16),(512,8,8)])