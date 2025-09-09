import math
import sys
import time
import torch

from utils.coco_utils import get_coco_api_from_dataset
from utils.coco_eval import CocoEvaluator
from utils import utils

from utils.metrics import *
import numpy as np 

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # try:
        #     loss_dict = model(images, targets)
        # except RuntimeError as exception:
        #     if "out of memory" in str(exception):
        #         print("WARNING: out of memory")
        #         if hasattr(torch.cuda, 'empty_cache'):
        #             torch.cuda.empty_cache()
        #     else:
        #         raise exception
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        torch.cuda.empty_cache()


def _get_iou_types(model):
    iou_types = ["bbox"]
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device,rag=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types,rag)
    # ct = False
    
    # for multi-label
    f1_scores = []
    jaccard_scores = []
    hamm_scores = []
    acc_scores = []

    for image, targets in metric_logger.log_every(data_loader, 100, header):

        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model_time = time.time()
        outputs = model(image)
        # for v in outputs:
        #     scores = []
        #     labels = []
        #     boxes = None
        #     for i in range(v['scores'].shape[0]):
        #         if v['scores'][i].item() >0.5:
        #             scores.append(v['scores'][i])
        #             labels.append(v['labels'][i])
        #             if boxes is None:
        #                 boxes = v['boxes'][i].unsqueeze(0)
        #             else:
        #                 boxes=torch.cat((boxes,v['boxes'][i].unsqueeze(0)))
        #     v['scores'] = torch.tensor(scores)
        #     v['labels'] = torch.tensor(labels)
        #     if boxes is None:
        #         ct = True
        #     v['boxes'] = boxes
        # if ct:
        #     ct = False
        #     continue

        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time


        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        for target, output in zip(targets, outputs):
            # filter labels with low scores in the output 
            filteredOutputLabels = filterOutput(output)
            f1 = f1_sampled(target['labels'].unsqueeze(0).cpu().numpy(), filteredOutputLabels.unsqueeze(0).cpu().numpy())
            f1_scores.append(f1)
            ja = jaccard_sampled(target['labels'].unsqueeze(0).cpu().numpy(), filteredOutputLabels.unsqueeze(0).cpu().numpy())
            jaccard_scores.append(ja)
            hamm = hamming_sampled(target['labels'].unsqueeze(0).cpu().numpy(), filteredOutputLabels.unsqueeze(0).cpu().numpy())
            hamm_scores.append(hamm)
            acc= acc_sampled(target['labels'].unsqueeze(0).cpu().numpy(), filteredOutputLabels.unsqueeze(0).cpu().numpy())
            acc_scores.append(acc)

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    # the mean of all f1 
    f1_mean = np.mean(f1_scores)
    ja_mean = np.mean(jaccard_scores)
    hamm_mean = np.mean(hamm_scores)
    acc_mean = np.mean(acc_scores)
    print('####### for multi-label metrics ######')
    print('Averaged f1_score:', f1_mean)
    print('Averaged jaccard_score:', ja_mean)
    print('Averaged hamming_loss:', hamm_mean)
    print('Averaged acc_score:', acc_mean)

    return coco_evaluator
