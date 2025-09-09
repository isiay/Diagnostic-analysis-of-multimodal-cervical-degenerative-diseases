import math
import sys
import time
import torch

from utils.coco_utils import get_coco_api_from_dataset
from utils.coco_eval import CocoEvaluator
from utils import utils


def _get_iou_types(model):
    iou_types = ["bbox"]
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, rag=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    #API 的作用就是为了提取标注文件中的信息, 使其分别用于各自的场景, 比如图像检测使用的边界框参数, 图像分割使用的 mask 参数, 人体姿态检测使用的关节点参数等
    iou_types = _get_iou_types(model)
    #iou指标
    coco_evaluator = CocoEvaluator(coco, iou_types, rag)
    #计算

    for image, targets in metric_logger.log_every(data_loader, 500, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model_time = time.time()
        outputs = model(image,targets)    #直接测试给定gt框的分类情况
        # print(outputs)
        # outputs = model(image)    #原来的写法

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if rag is not None:
        print('Range:', rag)
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
