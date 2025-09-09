import torch
import cv2
import numpy as np
import json
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from torch.utils.data import Dataset
from orientation_classifier import resnet


def cvt_BGR2Tensor(img):
    b, g, r = cv2.split(img)
    img = cv2.merge((r, g, b))
    img = torch.from_numpy(img)
    img = img.contiguous().permute(2, 0, 1)
    return img


class MedicalDataset(Dataset):

    def __init__(self, train, trans=None):
        self.train = train
        self.transforms = trans
        self.data = []
        if train:
            with open('/home/fym/code/MR/data/train_path_yuanfa_zhuanyi_three.txt', 'r') as path:
            # with open('/home/fym/code/MR/train_path.txt', 'r') as path:
                for x in path.readlines():
                    self.data.append(x[:-1])
            # with open('train_path_2.txt', 'r') as path:
            #     for x in path.readlines():
            #         self.data.append(x[:-1])
        else:
            # with open('yiyuan.txt', 'r') as path:
            # with open('test_path_shuffle.txt', 'r') as path:
            # with open('test_path.txt', 'r') as path:
            with open('/home/fym/code/MR/area_size/ct_yuanfa_shizhuangwei_area_320+.txt', 'r') as path:
                for x in path.readlines():
                    self.data.append(x[:-1])
                    # 这里path指的就是test_path.txt整个文件
                    # 然后path.readlines，读取文件的所有行
                    # 然后把每一行的除了最后一个元素的所有 添加到data里
                    # append 添加元素
            # with open('test_path_2.txt', 'r') as path:
            #     for x in path.readlines():
            #         self.data.append(x[:-1])

    def get_path(self, item):
        return self.data[item] + '.bmp', self.data[item] + '.json'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        bmp_path, json_path = self.data[item] + '.bmp', self.data[item] + '.json'
        img = cv2.imread(bmp_path)
        with open(json_path, 'r', encoding='gbk') as f:
            s = f.read()
            annotation = json.loads(s)['shapes']

        img = cvt_BGR2Tensor(img).float() / 255.0

        num = len(annotation)

        target = {}
        boxes = []
        for i in range(num):
            rect = annotation[i]['points']
            rect = np.array(rect).reshape((4,))
            if rect[0] + 5 <= rect[2] and rect[1] + 5 <= rect[3]:
                boxes.append(rect.tolist())  # 数据——列表
            else:
                num -= 1
        boxes = torch.tensor(boxes, dtype=torch.float)
        # print(boxes)
        # try:
        #     area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # except IndexError:
        #     print(json_path)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        labels = torch.ones((num,), dtype=torch.int64)
        image_id = torch.tensor([item])
        iscrowd = torch.zeros((num,), dtype=torch.uint8)
 
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        # print(img.size())

        return img, target


def init_mean_and_std():
    dataset = MedicalDataset(train=True)
    mean = torch.tensor([0., 0., 0.], dtype=torch.float64)
    std = torch.tensor([0., 0., 0.], dtype=torch.float64)
    num = torch.tensor([0.], dtype=torch.float64)
    for i in tqdm(range(len(dataset))):
        img, target = dataset[i]
        assert isinstance(img, torch.Tensor)
        img = img.double().reshape((3, -1))
        h, w = img.shape[-2:]
        num[0] += h * w
        mean += img.sum(dim=-1)
    mean = mean / num
    print('mean: ', mean)
    for i in tqdm(range(len(dataset))):
        img, target = dataset[i]
        assert isinstance(img, torch.Tensor)
        img = img.double().reshape((3, -1))
        img = (img - mean[:, None]) * (img - mean[:, None])
        std += img.sum(dim=-1)
    std = torch.sqrt(std / num)
    print('std: ', std)
# 计算数据集RGB三通道均值和方差

def show_dataset(idx: int):
    dataset = MedicalDataset(True)
    print('len dataset:', len(dataset))
    bmp_path, json_path = dataset.get_path(idx)
    img = cv2.imread(bmp_path)
    annotation = json.loads(open(json_path, encoding='gbk').read())['shapes']

    ax = plt.subplot(1, 1, 1)
    ax.imshow(img)
    for i in range(len(annotation)):
        rect = annotation[i]['points']
        rect = np.array(rect).reshape((4,))
        ax.add_patch(patches.Rectangle((rect[0], rect[1]),
                                       rect[2] - rect[0], rect[3] - rect[1],
                                       fill=False, edgecolor='r', linewidth=2))
    plt.show()


def test_json(json_path):
    with open(json_path, 'r', encoding='gbk') as f:
        s = f.read()
        annotation = json.loads(s)['shapes']
    num = len(annotation)
    for i in range(num):
        rect = annotation[i]['points']

        rect = np.array(rect)
        if rect.flatten().shape[0] != 4:
            num -= 1
            continue
        rect = rect.reshape((4,))
        if rect[0] + 5 > rect[2]:
            num -= 1
        elif rect[1] + 5 > rect[3]:
            num -= 1
    return num > 0
# 过滤错误标注

def init_data_path():
    root_paths = ['/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/李媛/MRI/',
                  '/home/fym/dataset/TOSHIBA_EXT//原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/刘剑芳/MRI/',
                  '/home/fym/dataset/TOSHIBA_EXT//原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/欧阳汉强/MRI/']

    with open('train_path.txt', 'w', encoding='utf-8') as f:
        for root_path in root_paths:
            for name1 in tqdm(os.listdir(root_path)):
                path1 = os.path.join(root_path, name1)
                for name2 in os.listdir(path1):
                    # if '轴位' not in name2 and '矢状位' not in name2:
                    #     continue
                    # if '轴位' not in name2:
                    #     continue
                    # if '矢状位' not in name2:
                    #     continue
                    path2 = os.path.join(path1, name2)
                    name_set = set(os.listdir(path2))
                    name_list = list(set([os.path.splitext(x)[0] for x in os.listdir(path2)]))
                    for name3 in name_list:
                        if name3 + '.bmp' in name_set and name3 + '.json' in name_set:
                            path3 = os.path.join(path2, name3)
                            if test_json(path3 + '.json'):
                                f.write(path3 + '\n')

    root_paths = ['/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/王春杰/MRI/',
                  '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/袁源/MRI/']

    with open('test_path.txt', 'w', encoding='utf-8') as f:
        for root_path in root_paths:
            for name1 in tqdm(os.listdir(root_path)):
                path1 = os.path.join(root_path, name1)
                for name2 in os.listdir(path1):
                    path2 = os.path.join(path1, name2)
                    # if '轴位' not in name2 and '矢状位' not in name2:
                    #     continue
                    # if '轴位' not in name2:
                    #     continue
                    # if '矢状位' not in name2:
                    #     continue
                    name_set = set(os.listdir(path2))
                    name_list = list(set([os.path.splitext(x)[0] for x in os.listdir(path2)]))
                    for name3 in name_list:
                        if name3 + '.bmp' in name_set and name3 + '.json' in name_set:
                            path3 = os.path.join(path2, name3)
                            if test_json(path3 + '.json'):
                                f.write(path3 + '\n')

def init_data_path_ct():
    root_paths = ['/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/李媛/CT骨窗/',
                  '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/李媛/CT软组织窗/',
                  '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/刘剑芳/CT骨窗/',
                  '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/刘剑芳/CT软组织窗/',
                  '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/欧阳汉强/CT骨窗/',
                  '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/欧阳汉强/CT软组织窗/']

    with open('/home/fym/code/MR/train_path_ct.txt', 'w', encoding='utf-8') as f:
        for root_path in root_paths:
            for name1 in tqdm(os.listdir(root_path)):
                path1 = os.path.join(root_path, name1)
                for name2 in os.listdir(path1):
                    path2 = os.path.join(path1, name2)
                    name_set = set(os.listdir(path2))
                    name_list = list(set([os.path.splitext(x)[0] for x in os.listdir(path2)]))
                    for name3 in name_list:
                        if name3 + '.bmp' in name_set and name3 + '.json' in name_set:
                            path3 = os.path.join(path2, name3)
                            if test_json(path3 + '.json'):
                                f.write(path3 + '\n')

    root_paths = ['/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/王春杰/CT骨窗/',
                  '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/王春杰/CT软组织窗/',
                  '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/袁源/CT骨窗/',
                  '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/袁源/CT软组织窗/']

    with open('/home/fym/code/MR/test_path_ct.txt', 'w', encoding='utf-8') as f:
        for root_path in root_paths:
            for name1 in tqdm(os.listdir(root_path)):
                path1 = os.path.join(root_path, name1)
                for name2 in os.listdir(path1):
                    path2 = os.path.join(path1, name2)
                    name_set = set(os.listdir(path2))
                    name_list = list(set([os.path.splitext(x)[0] for x in os.listdir(path2)]))
                    for name3 in name_list:
                        if name3 + '.bmp' in name_set and name3 + '.json' in name_set:
                            path3 = os.path.join(path2, name3)
                            if test_json(path3 + '.json'):
                                f.write(path3 + '\n')

def init_data_path_zhuanyiliu():  
    root_paths = ['/home/fym/dataset/tumer_2/转移瘤分位置/MRI2/李媛MRI2',
                  '/home/fym/dataset/tumer_2/转移瘤分位置/MRI2/欧阳汉强MRI2']
    with open('/home/fym/code/MR/train_path_zhuanyiliu.txt', 'w', encoding='utf-8') as f:
        for root_path in root_paths:
            for name1 in tqdm(os.listdir(root_path)):
                path1 = os.path.join(root_path, name1)
                for name2 in os.listdir(path1):
                    path2 = os.path.join(path1, name2)
                    name_set = set(os.listdir(path2))
                    name_list = list(set([os.path.splitext(x)[0] for x in os.listdir(path2)]))
                    for name3 in name_list:
                        if name3 + '.bmp' in name_set and name3 + '.json' in name_set:
                            path3 = os.path.join(path2, name3)
                            if test_json(path3 + '.json'):
                                f.write(path3 + '\n')
    root_paths = ['/home/fym/dataset/tumer_2/转移瘤分位置/MRI2/王春杰MRI2']
    with open('/home/fym/code/MR/test_path_zhuanyiliu.txt', 'w', encoding='utf-8') as f:
        for root_path in root_paths:
            for name1 in tqdm(os.listdir(root_path)):
                path1 = os.path.join(root_path, name1)
                for name2 in os.listdir(path1):
                    path2 = os.path.join(path1, name2)
                    name_set = set(os.listdir(path2))
                    name_list = list(set([os.path.splitext(x)[0] for x in os.listdir(path2)]))
                    for name3 in name_list:
                        if name3 + '.bmp' in name_set and name3 + '.json' in name_set:
                            path3 = os.path.join(path2, name3)
                            if test_json(path3 + '.json'):
                                f.write(path3 + '\n')

def init_data_path_ct_zhuanyiliu():  
    root_paths = ['/home/fym/dataset/tumer_2/转移瘤分位置/CT2/李媛CT2',
                  '/home/fym/dataset/tumer_2/转移瘤分位置/CT2/欧阳汉强CT2']
    with open('/home/fym/code/MR/train_path_ct_zhuanyiliu.txt', 'w', encoding='utf-8') as f:
        for root_path in root_paths:
            for name1 in tqdm(os.listdir(root_path)):
                path1 = os.path.join(root_path, name1)
                for name2 in os.listdir(path1):
                    path2 = os.path.join(path1, name2)
                    name_set = set(os.listdir(path2))
                    name_list = list(set([os.path.splitext(x)[0] for x in os.listdir(path2)]))
                    for name3 in name_list:
                        if name3 + '.bmp' in name_set and name3 + '.json' in name_set:
                            path3 = os.path.join(path2, name3)
                            if test_json(path3 + '.json'):
                                f.write(path3 + '\n')
    root_paths = ['/home/fym/dataset/tumer_2/转移瘤分位置/CT2/王春杰CT2']
    with open('/home/fym/code/MR/test_path_ct_zhuanyiliu.txt', 'w', encoding='utf-8') as f:
        for root_path in root_paths:
            for name1 in tqdm(os.listdir(root_path)):
                path1 = os.path.join(root_path, name1)
                for name2 in os.listdir(path1):
                    path2 = os.path.join(path1, name2)
                    name_set = set(os.listdir(path2))
                    name_list = list(set([os.path.splitext(x)[0] for x in os.listdir(path2)]))
                    for name3 in name_list:
                        if name3 + '.bmp' in name_set and name3 + '.json' in name_set:
                            path3 = os.path.join(path2, name3)
                            if test_json(path3 + '.json'):
                                f.write(path3 + '\n')

def init_data_path_2():  # {0: 轴位, 1: 矢状位, 2: 冠状位}
    with open('/home/fym/code/MR/orientation.json', 'r') as f:
        s = f.read()
        orientation_dict = json.loads(s)
    root_paths = ['/home/fym/dataset/tumer_2/刘剑芳/转移瘤第二次/转移瘤第二次整理/MRI2/李媛MRI2/',
                  '/home/fym/dataset/tumer_2/刘剑芳/转移瘤第二次/转移瘤第二次整理/MRI2/欧阳汉强MRI2/']
    with open('/home/fym/code/MR/train_path_2.txt', 'w', encoding='utf-8') as f:
        for root_path in root_paths:
            for name1 in tqdm(os.listdir(root_path)):
                path1 = os.path.join(root_path, name1)
                name_set = set(os.listdir(path1))
                name_list = list(set([os.path.splitext(x)[0] for x in os.listdir(path1)]))
                for name2 in name_list:
                    path2 = os.path.join(path1, name2)
                    o = orientation_dict.get(path2, -1)
                    if name2 + '.bmp' in name_set and name2 + '.json' in name_set:
                            # and (o == 0 or o == 2):
                        f.write(path2 + '\n')
    root_paths = ['/home/fym/dataset/tumer_2/刘剑芳/转移瘤第二次/转移瘤第二次整理/MRI2/王春杰MRI2/']
    with open('/home/fym/code/MR/test_path_2.txt', 'w', encoding='utf-8') as f:
        for root_path in root_paths:
            for name1 in tqdm(os.listdir(root_path)):
                path1 = os.path.join(root_path, name1)
                name_set = set(os.listdir(path1))
                name_list = list(set([os.path.splitext(x)[0] for x in os.listdir(path1)]))
                for name2 in name_list:
                    path2 = os.path.join(path1, name2)
                    o = orientation_dict.get(path2, -1)
                    if name2 + '.bmp' in name_set and name2 + '.json' in name_set:
                            # and (o == 0 or o == 2):
                        f.write(path2 + '\n')

def get_orientation(model, path, device):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = image[105:-105, ...]
    image = image / 255.0
    image = cv2.resize(image, (256, 256))
    image = torch.tensor(image, dtype=torch.float)
    image = image[None].repeat(3, 1, 1)
    image = image.to(device)
    with torch.no_grad():
        out = model(image[None])
        out = torch.argmax(out, dim=-1, keepdim=False).item()
        return out


def gen_orientation_json():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = resnet.resnet50(num_classes=3)
    model.load_state_dict(torch.load('/home/fym/code/MR/pth/classifier.pth'))
    model.to(device)

    root_paths = ['/home/fym/dataset/tumer_2/刘剑芳/转移瘤第二次/转移瘤第二次整理/MRI2/李媛MRI2/',
                  '/home/fym/dataset/tumer_2/刘剑芳/转移瘤第二次/转移瘤第二次整理/MRI2/欧阳汉强MRI2/',
                  '/home/fym/dataset/tumer_2/刘剑芳/转移瘤第二次/转移瘤第二次整理/MRI2/王春杰MRI2/']

    orientation_dict = dict()
    for root_path in root_paths:
        for name1 in tqdm(os.listdir(root_path)):
            path1 = os.path.join(root_path, name1)
            name_set = set(os.listdir(path1))
            name_list = list(set([os.path.splitext(x)[0] for x in os.listdir(path1)]))
            for name2 in name_list:
                path2 = os.path.join(path1, name2)
                if name2 + '.bmp' in name_set and name2 + '.json' in name_set:
                    if test_json(path2 + '.json'):
                        orientation_dict[path2] = get_orientation(model, path2 + '.bmp', device)

    with open('/home/fym/code/MR/orientation.json', 'w', encoding='utf-8') as f:
        s = json.dumps(orientation_dict)
        f.write(s)



if __name__ == '__main__':
    # gen_orientation_json()
    # init_data_path()
    mr2()
    # init_data_path_ct()
    # init_data_path_2()
    # init_data_path_zhuanyiliu()
    # init_data_path_ct_zhuanyiliu()
    # init_mean_and_std()
    print(len(MedicalDataset(train=True)))
    print(len(MedicalDataset(train=False)))
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=int, default=1)
    args = parser.parse_args()
    show_dataset(args.index)
    """
    pass
