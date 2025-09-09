import argparse
import torch
import torch.utils.data
import config
import os
from detection import generate_model
# from utils.engine import evaluate
from utils.engine1 import evaluate
from dataset.medical_dataset import MedicalDataset
from utils import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

def collate_fn(batch):
    return tuple(zip(*batch))


def eval(deformable: bool, model_path: str):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = config.num_classes_filter1  # classes

    test_dataset = MedicalDataset('eval', config.data_path, config.label_path)

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=3, shuffle=False,
                                                   num_workers=2, collate_fn=collate_fn)

    model = generate_model.faster_rcnn(num_classes=num_classes, deformable=deformable)
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    evaluate(model, test_data_loader, device, None)
    # evaluate(model, test_data_loader, device)


def init_argparser():
    parser = argparse.ArgumentParser(description='train faster rcnn')
    parser.add_argument('--deformable', type=bool, default=False)
    parser.add_argument('--save_path', type=str, default='/home/fym/code/MR2/pth/classifier_liang_e.pth')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_argparser()
    # eval(args.deformable, os.path.join('model_weights', args.save_path))
    eval(args.deformable, args.save_path)
