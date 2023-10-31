import os
import argparse
import os.path as osp
import torch.nn as nn
import se_resnet
import torch
import torch.optim as optim
import torch.utils.data as data
from torch.optim import lr_scheduler
import numpy as np
import time
import copy
from utils_cgw import Logger
from datasets import MedicalDataset


def train_model(model,dataloders, dataset_sizes, crtiation, optimizer, schedular, device, args):
   
    begin_time = time.time()
    best_acc = 0.0
    bestacc_epoch = -2
    run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
    log_dir = os.path.join('./new_model_logger')
    log_dir = os.path.join(log_dir, args.dataset)
  

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log = os.path.join(log_dir,args.model_name+'_'+run_id+".log")
    mylogger = Logger(args.model_name+'_'+run_id, log)
    mylogger.info(args)

    for epoch in range(0,args.epoch):
        print("-*-" * 20)
        for phase in ['Train', 'eval']:
            if phase=='Train':                
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            zlevel_c = 0.0
            level_c = 0.0

            for images, labels,pos,bmg_path in dataloders[phase]:

                images.to(device)
                labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='Train'):
                   
                    opt = model(images.cuda())
                   
                    labels = labels.cuda()
                    loss = crtiation(opt, labels)
                    if phase=='Train':
                        loss.backward()
                        optimizer.step()   
                        
                running_loss += loss.item()*images.size(0)
               
                for row in (opt.ge(0.5) == labels):
                    if row[0] == True:
                        zlevel_c+=1
                    if row[1] == True:
                        level_c+=1
                    
            if phase == 'Train':
                schedular.step()
            epoch_loss = running_loss/dataset_sizes[phase]
            zlevel_acc = zlevel_c/dataset_sizes[phase]
            level_acc = level_c/dataset_sizes[phase]
           
            info = 'epoch = {}, Phase = {}, Loss = {:.4f}, zlevel_ACC:{:.4f}, level_ACC:{:.4f}'.format(epoch, phase, epoch_loss, zlevel_acc*100, level_acc*100)
          

            if phase == 'eval' and (zlevel_acc>best_acc):
                # Upgrade the weights 
                best_acc=zlevel_acc 
                bestacc_epoch  = epoch
                best_weights = copy.deepcopy(model.state_dict())
                # Save model_dict               
                if not os.path.exists(args.path):
                    os.makedirs(args.path)
                torch.save(model.state_dict(),osp.join(args.path,"{}_{}_{}_epoch{}.pth".format(args.model_name,run_id,args.dataset,epoch)))
            mylogger.info(info)

    time_elapes = time.time() - begin_time
   
    info = 'Training Complete in {:.0f}m {:0f}s'.format(
        time_elapes // 60, time_elapes % 60
    )
    mylogger.info(info)
    print(info)

    info = 'Best Val ACC: {:}  bestacc_epoch: {}\n'.format(best_acc,bestacc_epoch)
    mylogger.info(info)
    model.load_state_dict(best_weights)

    return model


def main(args):
     
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
   
    model = se_resnet.resnet50(pretrained=True)
  
    numfits = model.fc.in_features
    num_classes = 2
    model.fc = nn.Linear(numfits,num_classes)

    device = torch.device('cuda')
    model.to(device)

    loss_criterion = nn.BCELoss()
   
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = 0.9, weight_decay= args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = args.decay_step,gamma=0.1)

    
    train_datasets = MedicalDataset('train')
    eval_datasets = MedicalDataset('eval')

    train_loader = data.DataLoader(train_datasets, batch_size=args.batchsize, shuffle=True,num_workers=10)
    eval_loader = data.DataLoader(eval_datasets, batch_size=args.batchsize, shuffle=True,num_workers=10)

    dataloder = {'Train':train_loader, 'eval': eval_loader}
    dataset_sizes = {'Train': len(train_datasets), 'eval':len(eval_datasets)} 

    print('Strat training')
    print('dataset_sizes ',args.dataset,dataset_sizes)
    
    train_model(model, dataloder, dataset_sizes, loss_criterion, optimizer, exp_lr_scheduler, device, args)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default='mri_all', help='Train Dataset')
    parser.add_argument('--batchsize', type = int, default=32)
    parser.add_argument('--epoch', type = int, default=13)
    parser.add_argument('--lr',type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type = float, default=0.0005)
    parser.add_argument('--decay_step', type = int, default=1)
    parser.add_argument('--convolution_name', default =None, help='Define the Conv we use')
    parser.add_argument('--path', default='./new_pth',type= str)
    parser.add_argument('--cuda_id', default='1')
    parser.add_argument('--model_name',default='MR_disc')
    parser.add_argument('--optimizer_name',default='SGD')
    parser.add_argument('--cut',default='none')
    parser.add_argument('--loadmodelpath',default='none')
    args = parser.parse_args()
    main(args)
