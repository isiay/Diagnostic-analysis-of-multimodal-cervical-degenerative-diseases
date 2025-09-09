# ------------------------------------------------------------------------------
# Adapted from https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
# Original licence: Copyright (c) Microsoft, under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import dsntnn
import numpy as np
import pdb

from ..builder import LOSSES


def _make_input(t, requires_grad=False, device=torch.device('cpu')):
    """Make zero inputs for AE loss.

    Args:
        t (torch.Tensor): input
        requires_grad (bool): Option to use requires_grad.
        device: torch device

    Returns:
        torch.Tensor: zero input.
    """
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    inp = inp.to(device)
    return inp


@LOSSES.register_module()
class HeatmapLoss(nn.Module):
    """Accumulate the heatmap loss for each image in the batch.

    Args:
        supervise_empty (bool): Whether to supervise empty channels.
    """

    def __init__(self, supervise_empty=True):
        super().__init__()
        self.supervise_empty = supervise_empty

    def forward(self, pred, gt, mask):
        """Forward function.

        Note:
            - batch_size: N
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            pred (torch.Tensor[N,K,H,W]):heatmap of output.
            gt (torch.Tensor[N,K,H,W]): target heatmap.
            mask (torch.Tensor[N,H,W]): mask of target.
        """
        assert pred.size() == gt.size(
        ), f'pred.size() is {pred.size()}, gt.size() is {gt.size()}'

        if not self.supervise_empty:
            empty_mask = (gt.sum(dim=[2, 3], keepdim=True) > 0).float()
            loss = ((pred - gt)**2) * empty_mask.expand_as(
                pred) * mask[:, None, :, :].expand_as(pred)
        else:
            loss = ((pred - gt)**2) * mask[:, None, :, :].expand_as(pred)
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        return loss


@LOSSES.register_module()
class MultiTaskLoss(nn.Module):
    """Accumulate the multi-tasking loss for each image in the batch. Implemented by WKA.
    Find the corresponding keypoint id in '/home/myy/mmpose/关键点编号与骨头的对应关系.png'

    Args:
        supervise_empty (bool): Whether to supervise empty channels.
    """

    def __init__(self, supervise_empty=True):
        # print('[INFO] MultiTaskLoss initializing. Implemented by wka')
        super().__init__()
        self.supervise_empty = supervise_empty

    def cobb_loss(self, pred, gt):
        """calculate the cobb loss

        Args:
            pred (torch.Tensor[N,24,2]): dsnt result of the pred heatmap
            gt   (torch.Tensor[N,24,2]): dsnt result of the gt heatmap
        """
        assert pred.shape[1:] == torch.Size([24, 2]
                                            ), f'pred.shape and gt.shape are {pred.shape}'

        loss = (gt[:, [1, 2, 21, 22], :] - pred[:, [1, 2, 21, 22], :])**2
        loss = loss.mean(2).mean(1)  # torch.Size([8])
        # print(f'cobb_loss \t= {loss}')
        return loss

    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    def intersection(L1, L2):
        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x, y
        else:
            # return False
            return 0, 0

    def sva_loss(self, pred, gt):
        """calculate the sva loss

        Args:
            pred (torch.Tensor[N,24,2]): dsnt result of the pred heatmap
            gt   (torch.Tensor[N,24,2]): dsnt result of the gt heatmap
        """

        loss = []
        for i in range(len(gt.shape[0])):
            inter_point_gt = self.intersection(
                self.line(gt[i, 20], gt[i, 22]), self.line(gt[i, 23], gt[i, 21]))
            inter_point_pred = self.intersection(
                self.line(pred[i, 20], pred[i, 22]), self.line(pred[i, 23], pred[i, 21]))
            
            # need left actually # whether the last index is 1 or 0?
            sva_gt = gt[i, 3, 1] - inter_point_gt[1]   
            sva_pred = pred[i, 3, 1] - inter_point_pred[1]
            loss.append((sva_pred - sva_gt)**2)
        loss = torch.stack(loss)
        print(f'sva_loss \t= {loss}')
        return loss


    def spcd_loss(self, pred, gt):
        """calculate the spcd(椎体矢状径) loss

        Args:
            pred (torch.Tensor[N,24,2]): dsnt result of the pred heatmap
            gt   (torch.Tensor[N,24,2]): dsnt result of the gt heatmap
        """
        gt = ((gt[:, [19, 15, 11, 7, 3]] + gt[:, [18, 14, 10, 6, 2]]) -
                    (gt[:, [17, 13, 19, 5, 1]] + gt[:, [16, 12, 8,  4, 0]])) / 2
        gt = (gt**2).sum(dim=2) # torch.Size([8, 5])
        
        pred = ((pred[:, [19, 15, 11, 7, 3]] + pred[:, [18, 14, 10, 6, 2]]) -
                    (pred[:, [17, 13, 19, 5, 1]] + pred[:, [16, 12, 8,  4, 0]])) / 2
        pred = (pred**2).sum(dim=2)  # torch.Size([8, 5])

        loss = (pred-gt) ** 2
        loss = loss.mean(dim=1) # torch.Size([8])

        # print(f'spcd_loss \t= {loss}')
        return loss

    def forward(self, pred, gt, mask):
        """Forward function.

        Note:
            - batch_size: N
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            pred (torch.Tensor[N,K,H,W]):heatmap of output.
            gt (torch.Tensor[N,K,H,W]): target heatmap.
            mask (torch.Tensor[N,H,W]): mask of target.
        """
        assert pred.size() == gt.size(
        ), f'pred.size() is {pred.size()}, gt.size() is {gt.size()}'

        assert torch.isnan(
            gt).any() == False, f'There is nan in gt heatmaps. gt = {gt}'
        assert torch.isnan(pred).any(
        ) == False, f'There is nan in pred heatmaps. pred = {pred}'

        # Tensor.view() requires that the tensor is contiguous. Added by wka
        pred = pred.contiguous()

        dsnt_gt = dsntnn.dsnt(gt)
        dsnt_pred = dsntnn.dsnt(pred)

        ratio = [1e-6, 1e-6, 1e-8]

        loss = ratio[0] * self.cobb_loss(dsnt_pred, dsnt_gt) + \
            ratio[1] * self.sva_loss(dsnt_pred, dsnt_gt) + \
            ratio[2] * self.spcd_loss(dsnt_pred, dsnt_gt)
        loss = loss.to(device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"))

        # print('org loss=', loss)
        # assert torch.isnan(loss).any(
        # ) == False, f'There is nan in org loss. org loss = {loss}'

        loss = loss
        return loss


@LOSSES.register_module()
class AELoss(nn.Module):
    """Associative Embedding loss.

    `Associative Embedding: End-to-End Learning for Joint Detection and
    Grouping <https://arxiv.org/abs/1611.05424v2>`_.
    """

    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

    def singleTagLoss(self, pred_tag, joints):
        """Associative embedding loss for one image.

        Note:
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            pred_tag (torch.Tensor[KxHxW,1]): tag of output for one image.
            joints (torch.Tensor[M,K,2]): joints information for one image.
        """
        tags = []
        pull = 0
        for joints_per_person in joints:
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp))**2)

        num_tags = len(tags)
        if num_tags == 0:
            return (
                _make_input(torch.zeros(1).float(), device=pred_tag.device),
                _make_input(torch.zeros(1).float(), device=pred_tag.device))
        elif num_tags == 1:
            return (_make_input(
                torch.zeros(1).float(), device=pred_tag.device), pull)

        tags = torch.stack(tags)

        size = (num_tags, num_tags)
        A = tags.expand(*size)
        B = A.permute(1, 0)

        diff = A - B

        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_tags
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unknown ae loss type')

        push_loss = push / ((num_tags - 1) * num_tags) * 0.5
        pull_loss = pull / (num_tags)

        return push_loss, pull_loss

    def forward(self, tags, joints):
        """Accumulate the tag loss for each image in the batch.

        Note:
            - batch_size: N
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            tags (torch.Tensor[N,KxHxW,1]): tag channels of output.
            joints (torch.Tensor[N,M,K,2]): joints information.
        """
        pushes, pulls = [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)
        return torch.stack(pushes), torch.stack(pulls)


@LOSSES.register_module()
class MultiLossFactory(nn.Module):
    """Loss for bottom-up models. Edited by WKA.

    Args:
        num_joints (int): Number of keypoints.
        num_stages (int): Number of stages.
        ae_loss_type (str): Type of ae loss.
        with_ae_loss (list[bool]): Use ae loss or not in multi-heatmap.
        push_loss_factor (list[float]):
            Parameter of push loss in multi-heatmap.
        pull_loss_factor (list[float]):
            Parameter of pull loss in multi-heatmap.
        with_heatmap_loss (list[bool]):
            Use heatmap loss or not in multi-heatmap.
        heatmaps_loss_factor (list[float]):
            Parameter of heatmap loss in multi-heatmap.
        supervise_empty (bool): Whether to supervise empty channels.
    """

    def __init__(self,
                 num_joints,
                 num_stages,
                 ae_loss_type,
                 with_ae_loss,
                 push_loss_factor,
                 pull_loss_factor,
                 with_heatmaps_loss,
                 heatmaps_loss_factor,
                 supervise_empty=True):
        super().__init__()

        assert isinstance(with_heatmaps_loss, (list, tuple)), \
            'with_heatmaps_loss should be a list or tuple'
        assert isinstance(heatmaps_loss_factor, (list, tuple)), \
            'heatmaps_loss_factor should be a list or tuple'
        assert isinstance(with_ae_loss, (list, tuple)), \
            'with_ae_loss should be a list or tuple'
        assert isinstance(push_loss_factor, (list, tuple)), \
            'push_loss_factor should be a list or tuple'
        assert isinstance(pull_loss_factor, (list, tuple)), \
            'pull_loss_factor should be a list or tuple'

        self.num_joints = num_joints
        self.num_stages = num_stages
        self.ae_loss_type = ae_loss_type
        self.with_ae_loss = with_ae_loss
        self.push_loss_factor = push_loss_factor
        self.pull_loss_factor = pull_loss_factor
        self.with_heatmaps_loss = with_heatmaps_loss
        self.heatmaps_loss_factor = heatmaps_loss_factor

        self.heatmaps_loss = \
            nn.ModuleList(
                [
                    HeatmapLoss(supervise_empty)
                    if with_heatmaps_loss else None
                    for with_heatmaps_loss in self.with_heatmaps_loss
                ]
            )

        self.ae_loss = \
            nn.ModuleList(
                [
                    AELoss(self.ae_loss_type) if with_ae_loss else None
                    for with_ae_loss in self.with_ae_loss
                ]
            )

        # implemented by WKA
        self.multitask_loss = \
            nn.ModuleList(
                [
                    MultiTaskLoss() if with_heatmaps_loss else None
                    for with_heatmaps_loss in self.with_heatmaps_loss
                ]
            )

    def forward(self, outputs, heatmaps, masks, joints):
        """Forward function to calculate losses.

        Note:
            - batch_size: N
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K
            - output_channel: C C=2K if use ae loss else K

        Args:
            outputs (list(torch.Tensor[N,C,H,W])): outputs of stages.
            heatmaps (list(torch.Tensor[N,K,H,W])): target of heatmaps.
            masks (list(torch.Tensor[N,H,W])): masks of heatmaps.
            joints (list(torch.Tensor[N,M,K,2])): joints of ae loss.
        """
        heatmaps_losses = []
        push_losses = []
        pull_losses = []
        multitask_losses = []
        for idx in range(len(outputs)):
            offset_feat = 0
            '''
            if self.heatmaps_loss[idx]:
                heatmaps_pred = outputs[idx][:, :self.num_joints]
                offset_feat = self.num_joints
                heatmaps_loss = self.heatmaps_loss[idx](heatmaps_pred,
                                                        heatmaps[idx],
                                                        masks[idx])
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmaps_losses.append(heatmaps_loss)
            else:
                heatmaps_losses.append(None)
            '''

            # implemented by wka
            if self.heatmaps_loss[idx]:
                heatmaps_pred = outputs[idx][:, :self.num_joints]
                offset_feat = self.num_joints

                heatmaps_loss = self.heatmaps_loss[idx](heatmaps_pred,
                                                        heatmaps[idx],
                                                        masks[idx])
                multitask_loss = self.multitask_loss[idx](heatmaps_pred,
                                                          heatmaps[idx],
                                                          masks[idx])
                # print('heatmaps_loss=', heatmaps_loss)
                # print('multitask_loss=', multitask_loss)
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                multitask_loss = multitask_loss * \
                    self.heatmaps_loss_factor[idx]

                heatmaps_losses.append(heatmaps_loss)
                multitask_losses.append(multitask_loss)
            else:
                heatmaps_losses.append(None)
                multitask_losses.append(None)

            if self.ae_loss[idx]:
                tags_pred = outputs[idx][:, offset_feat:]
                batch_size = tags_pred.size()[0]
                tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)

                push_loss, pull_loss = self.ae_loss[idx](tags_pred,
                                                         joints[idx])
                push_loss = push_loss * self.push_loss_factor[idx]
                pull_loss = pull_loss * self.pull_loss_factor[idx]

                push_losses.append(push_loss)
                pull_losses.append(pull_loss)
            else:
                push_losses.append(None)
                pull_losses.append(None)

        assert type(heatmaps_losses) == type(
            multitask_losses), f'heatmaps_losses {type(heatmaps_losses)} while multitask_losses {type(multitask_losses)}'
        assert type(heatmaps_losses[0]) == type(
            multitask_losses[0]), f'heatmaps_losses[0] {type(heatmaps_losses[0])} while multitask_losses[0] {type(multitask_losses[0])}'
        return heatmaps_losses, push_losses, pull_losses, multitask_losses


# backup of the original MultiLossFactory
'''
@LOSSES.register_module()
class MultiLossFactory(nn.Module):
    """Loss for bottom-up models.

    Args:
        num_joints (int): Number of keypoints.
        num_stages (int): Number of stages.
        ae_loss_type (str): Type of ae loss.
        with_ae_loss (list[bool]): Use ae loss or not in multi-heatmap.
        push_loss_factor (list[float]):
            Parameter of push loss in multi-heatmap.
        pull_loss_factor (list[float]):
            Parameter of pull loss in multi-heatmap.
        with_heatmap_loss (list[bool]):
            Use heatmap loss or not in multi-heatmap.
        heatmaps_loss_factor (list[float]):
            Parameter of heatmap loss in multi-heatmap.
        supervise_empty (bool): Whether to supervise empty channels.
    """

    def __init__(self,
                 num_joints,
                 num_stages,
                 ae_loss_type,
                 with_ae_loss,
                 push_loss_factor,
                 pull_loss_factor,
                 with_heatmaps_loss,
                 heatmaps_loss_factor,
                 supervise_empty=True):
        super().__init__()

        assert isinstance(with_heatmaps_loss, (list, tuple)), \
            'with_heatmaps_loss should be a list or tuple'
        assert isinstance(heatmaps_loss_factor, (list, tuple)), \
            'heatmaps_loss_factor should be a list or tuple'
        assert isinstance(with_ae_loss, (list, tuple)), \
            'with_ae_loss should be a list or tuple'
        assert isinstance(push_loss_factor, (list, tuple)), \
            'push_loss_factor should be a list or tuple'
        assert isinstance(pull_loss_factor, (list, tuple)), \
            'pull_loss_factor should be a list or tuple'

        self.num_joints = num_joints
        self.num_stages = num_stages
        self.ae_loss_type = ae_loss_type
        self.with_ae_loss = with_ae_loss
        self.push_loss_factor = push_loss_factor
        self.pull_loss_factor = pull_loss_factor
        self.with_heatmaps_loss = with_heatmaps_loss
        self.heatmaps_loss_factor = heatmaps_loss_factor

        self.heatmaps_loss = \
            nn.ModuleList(
                [
                    HeatmapLoss(supervise_empty)
                    if with_heatmaps_loss else None
                    for with_heatmaps_loss in self.with_heatmaps_loss
                ]
            )

        self.ae_loss = \
            nn.ModuleList(
                [
                    AELoss(self.ae_loss_type) if with_ae_loss else None
                    for with_ae_loss in self.with_ae_loss
                ]
            )

    def forward(self, outputs, heatmaps, masks, joints):
        """Forward function to calculate losses.

        Note:
            - batch_size: N
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K
            - output_channel: C C=2K if use ae loss else K

        Args:
            outputs (list(torch.Tensor[N,C,H,W])): outputs of stages.
            heatmaps (list(torch.Tensor[N,K,H,W])): target of heatmaps.
            masks (list(torch.Tensor[N,H,W])): masks of heatmaps.
            joints (list(torch.Tensor[N,M,K,2])): joints of ae loss.
        """
        heatmaps_losses = []
        push_losses = []
        pull_losses = []
        for idx in range(len(outputs)):
            offset_feat = 0
            if self.heatmaps_loss[idx]:
                heatmaps_pred = outputs[idx][:, :self.num_joints]
                offset_feat = self.num_joints
                heatmaps_loss = self.heatmaps_loss[idx](heatmaps_pred,
                                                        heatmaps[idx],
                                                        masks[idx])
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmaps_losses.append(heatmaps_loss)
            else:
                heatmaps_losses.append(None)

            if self.ae_loss[idx]:
                tags_pred = outputs[idx][:, offset_feat:]
                batch_size = tags_pred.size()[0]
                tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)

                push_loss, pull_loss = self.ae_loss[idx](tags_pred,
                                                         joints[idx])
                push_loss = push_loss * self.push_loss_factor[idx]
                pull_loss = pull_loss * self.pull_loss_factor[idx]

                push_losses.append(push_loss)
                pull_losses.append(pull_loss)
            else:
                push_losses.append(None)
                pull_losses.append(None)

        return heatmaps_losses, push_losses, pull_losses
'''
