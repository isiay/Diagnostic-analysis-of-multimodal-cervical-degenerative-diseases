import torch
import torch.utils.data
import argparse
import platform
import os
import config
import math
from detection import generate_model
from utils.engine import evaluate
from utils.lr_policy import WarmUpPolyLR
from dataset.medical_dataset_random import MedicalDataset
from utils import utils
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def train_one_epoch(model, optimizer, data_loader, lr_policy, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # print(targets[0]['labels'])
        loss_dict, detections = model(images, targets)
        # print(detections)
        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in loss_dict]
        # print(outputs)
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
        lr_policy.step()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def collate_fn(batch):
    return tuple(zip(*batch))


def train(deformable: bool, model_path: str):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    num_classes = config.num_classes_filter1  # classes

    train_dataset = MedicalDataset('train', config.label_path)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True,
                                                    num_workers=2, collate_fn=collate_fn)

    test_dataset = MedicalDataset('eval', config.label_path)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False,
                                                   num_workers=2, collate_fn=collate_fn)

    model = generate_model.faster_rcnn(num_classes=num_classes, deformable=deformable)

    # print(model)
    model.to(device)

    num_epochs = config.num_epochs
    # lr = config.base_lr
    lr = 1e-2

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=config.momentum, weight_decay=config.weight_decay)

    # lr_policy = WarmUpPolyLR(optimizer, lr, config.lr_power, 
    #                         config.num_epochs * len(train_data_loader), 
    #                         config.warmup_epoch * len(train_data_loader))
    lr_policy  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1080, gamma=0.5)
    # model.load_state_dict(torch.load("/home/fym/code/MR2/pth/detection_mr_classifier.pth"))
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_data_loader, lr_policy, device, epoch, 10)
        
        # if epoch % 5 == 4:
        #     evaluate(model, test_data_loader, device, None)
        torch.save(model.state_dict(), model_path)
        print('Finish')
    
    torch.save(model.state_dict(), model_path)
    print("finish")


def init_argparser():
    parser = argparse.ArgumentParser(description='train faster rcnn')
    parser.add_argument('--deformable', type=bool, default=False)
    parser.add_argument('--save_path', type=str, default='/home/myy/jingzhui/MRI/MRI_pth_detection/classfier_2_kong2_v9.pth')
    args = parser.parse_args()
    return args 


if __name__ == '__main__':
    args = init_argparser()
    train(args.deformable, os.path.join('model_weights', args.save_path))

