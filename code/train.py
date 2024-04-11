import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import matplotlib.pylab as plt
running_loss_final = 0

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    # print(n,c,h,w)
    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    # print(input.shape,target.shape)
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0] # 262144 #input = 2*256*256*2
    input = input.view(-1, c)
    mask = target >= 0
    target = target[mask]
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def cross_entropy2d_edge(input, target, reduction='sum'):
    assert (input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()
    # ing = ((torch.gt(target, 0) & torch.lt(target, 1))).float()
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)


class Trainer(object):
    def __init__(self, cuda, model_depth, model_baseline, model_fusion,  optimizer_depth, optimizer_baseline, optimizer_ladder,
                 train_loader, max_iter, snapshot, outpath, sshow, size_average=False):
        self.cuda = cuda
        self.model_depth = model_depth
        self.model_baseline = model_baseline
        self.model_fusion = model_fusion
        self.optim_depth = optimizer_depth
        self.optim_baseline = optimizer_baseline
        self.optim_ladder = optimizer_ladder
        self.train_loader = train_loader
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.snapshot = snapshot
        self.outpath = outpath
        self.sshow = sshow
        self.size_average = size_average

    def train_epoch(self):
        for batch_idx, (img, mask, depth, edge) in enumerate(self.train_loader):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration
            if self.iteration >= self.max_iter:
                break
            if self.cuda:
                img, mask, depth, edge = img.cuda(), mask.cuda(), depth.cuda(), edge.cuda()
                img, mask, depth, edge = Variable(img), Variable(mask), Variable(depth), Variable(edge)
            n, c, h, w = img.size()  # batch_size, channels, height, weight

            self.optim_depth.zero_grad()
            self.optim_baseline.zero_grad()
            self.optim_ladder.zero_grad()

            global running_loss_final
            depth = depth.view(n, 1, h, w).repeat(1, c, 1, 1)
            # depth = depth.view(n, 1, h, w)
            mask = mask.view(n, 1, h, w)
            edge = edge.view(n, 1, h, w)
            d2, d3, d4, d5 = self.model_depth(depth)
            h2, h3, h4, h5 = self.model_baseline(img)
            p2, p3, p4, p5, p = self.model_fusion(img,h2, h3, h4, h5, d2, d3, d4, d5)

            mask = mask.to(torch.float32)
            edge = edge.to(torch.float32)
            loss_e = cross_entropy2d_edge(p2, edge)
            loss_0 = F.binary_cross_entropy_with_logits(p3, mask, reduction='sum')
            loss_1 = F.binary_cross_entropy_with_logits(p4, mask, reduction='sum')
            loss_2 = F.binary_cross_entropy_with_logits(p5, mask, reduction='sum')
            loss_3 = F.binary_cross_entropy_with_logits(p , mask, reduction='sum')


            loss_all = loss_e + loss_0 + loss_1 + loss_2 + loss_3
            running_loss_final = loss_all.item()

            if iteration % self.sshow == (self.sshow - 1):
                print('\n [%3d, %6d,   RGB-D Net loss: %.3f]' % (
                self.epoch + 1, iteration + 1, running_loss_final / (n * self.sshow) ))

                running_loss_final = 0.0

            if iteration <= 200000:
                if iteration % self.snapshot == (self.snapshot - 1):
                    savename_depth = ('%s/depth_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_depth.state_dict(), savename_depth)
                    print('save: (snapshot: %d)' % (iteration + 1))

                    savename_baseline = ('%s/baseline_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_baseline.state_dict(), savename_baseline)
                    print('save: (snapshot: %d)' % (iteration + 1))

                    savename_ladder = ('%s/ladder_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_fusion.state_dict(), savename_ladder)
                    print('save: (snapshot: %d)' % (iteration + 1))
            else:

                if iteration % 10000 == (10000 - 1):
                    savename_depth = ('%s/depth_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_depth.state_dict(), savename_depth)
                    print('save: (snapshot: %d)' % (iteration + 1))

                    savename_baseline = ('%s/baseline_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_baseline.state_dict(), savename_baseline)
                    print('save: (snapshot: %d)' % (iteration + 1))

                    savename_ladder = ('%s/ladder_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_fusion.state_dict(), savename_ladder)
                    print('save: (snapshot: %d)' % (iteration + 1))

            if (iteration + 1) == self.max_iter:
                savename_depth = ('%s/depth_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                torch.save(self.model_depth.state_dict(), savename_depth)
                print('save: (snapshot: %d)' % (iteration + 1))

                savename_baseline = ('%s/baseline_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                torch.save(self.model_baseline.state_dict(), savename_baseline)
                print('save: (snapshot: %d)' % (iteration + 1))

                savename_ladder = ('%s/ladder_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                torch.save(self.model_fusion.state_dict(), savename_ladder)
                print('save: (snapshot: %d)' % (iteration + 1))

            loss_all.backward()
            self.optim_depth.step()
            self.optim_baseline.step()
            self.optim_ladder.step()

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))

        for epoch in range(max_epoch):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
