import torch
import random
import os
import json
import numpy as np
import pandas as pd
import config
from tqdm import tqdm
from dataset.medical_dataset import MedicalDataset


data_path = '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp'
                    # '/home/fym/dataset/tumer_2/转移瘤分位置']
label_path = '/home/fym/dataset/TOSHIBA_EXT/label.csv'


def read_json(save_path, encoding='utf8'):
    with open(save_path, 'r', encoding=encoding) as f:
        content = f.read()
        content = json.loads(content)
        return content
    
def filter_data(data):
    print('[filter dataset]')
    filtered_data = []
    for item in tqdm(range(len(data))):
        json_path = data[item][1] + '.json'
        annotation = read_json(json_path, encoding='gbk')['shapes']
        num = len(annotation)
        tag = True
        if num == 0:
            tag = False
        for i in range(num):
            rect = annotation[i]['points']
            rect = np.array(rect).reshape((-1,))
            if rect.shape[0] != 4 or (not (rect[0] < rect[2] and rect[1] < rect[3])):
                tag = False
                break
        if tag:
            filtered_data.append(data[item])
    print('[Finish]')
    return filtered_data


def gen_dataset_path():
    np.random.seed(123098)
    patients = np.array(range(0, 624))
    np.random.shuffle(patients)
    train_set = set(patients[:400].tolist())
    test_set = set(patients[400:].tolist())

    train_data = []  # (patient id, data path)
    test_data = []

    p1 = data_path
    for f1 in sorted(os.listdir(p1)):
        p2 = os.path.join(p1, f1, 'MRI')
        for f2 in tqdm(sorted(os.listdir(p2))):
            patient_id = int(f2[7:12])  # Patient
            p3 = os.path.join(p2, f2)
            for f3 in sorted(os.listdir(p3)):
                p4 = os.path.join(p3, f3)
                bmps = set()
                jsons = set()
                for f4 in sorted(os.listdir(p4)):
                    if 'bmp' in f4:
                        bmps.add(f4.split('.')[0])
                    elif 'json' in f4:
                        jsons.add(f4.split('.')[0])
                files = list(bmps & jsons)
                if patient_id in train_set:
                    for f in files:
                        train_data.append((patient_id, os.path.join(p4, f)))
                else:
                    for f in files:
                        test_data.append((patient_id, os.path.join(p4, f)))
    train_data = filter_data(train_data)
    test_data = filter_data(test_data)
    content = {
        'train_data': train_data,
        'test_data': test_data
    }
    with open('/home/fym/code/MR2/data.json', 'w') as f:
        content = json.dumps(content)
        f.write(content)

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

def init_data_path():
    root_paths = ['/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/李媛/MRI/',
                  '/home/fym/dataset/TOSHIBA_EXT//原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/刘剑芳/MRI/',
                  '/home/fym/dataset/TOSHIBA_EXT//原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/欧阳汉强/MRI/',
                  '/home/fym/dataset/tumer_2/转移瘤分位置/MRI2/李媛MRI2/',
                  '/home/fym/dataset/tumer_2/转移瘤分位置/MRI2/欧阳汉强MRI2/']

    train_data = []  # (patient id, data path)
    test_data = []
    str_1 = "原发"
    str_2 = "转移"
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
                                if str_1 in root_path:
                                    label_ID =1
                                    train_data.append((label_ID, path3))
                                elif str_2 in root_path:
                                    label_ID =2
                                    train_data.append((label_ID, path3))
    root_paths = ['/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/王春杰/MRI/',
                '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/袁源/MRI/',
                '/home/fym/dataset/tumer_2/转移瘤分位置/MRI2/王春杰MRI2/']

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
                            if str_1 in root_path:
                                label_ID =1
                                test_data.append((label_ID, path3))
                            elif str_2 in root_path:
                                label_ID =2
                                test_data.append((label_ID, path3))

    train_data = filter_data(train_data)
    test_data = filter_data(test_data)
    content = {
        'train_data': train_data,
        'test_data': test_data
    }
    with open('./data_mr_yuanfa.json', 'w') as f:
        content = json.dumps(content)
        f.write(content)

    # root_paths = ['/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/李媛/MRI/',
    #               '/home/fym/dataset/TOSHIBA_EXT//原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/刘剑芳/MRI/',
    #               '/home/fym/dataset/TOSHIBA_EXT//原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/欧阳汉强/MRI/']
    # with open('/home/fym/code/MR2/train_path.json', 'w', encoding='utf-8') as f:
    #     for root_path in root_paths:
    #         for name1 in tqdm(os.listdir(root_path)):
    #             path1 = os.path.join(root_path, name1)
    #             for name2 in os.listdir(path1):
    #                 path2 = os.path.join(path1, name2)
    #                 name_set = set(os.listdir(path2))
    #                 name_list = list(set([os.path.splitext(x)[0] for x in os.listdir(path2)]))
    #                 for name3 in name_list:
    #                     if name3 + '.bmp' in name_set and name3 + '.json' in name_set:
    #                         path3 = os.path.join(path2, name3)
    #                         if test_json(path3 + '.json'):
    #                             f.write(path3 + '\n')

    # root_paths = ['/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/王春杰/MRI/',
    #               '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/袁源/MRI/']

    # with open('./test_path.json', 'w', encoding='utf-8') as f:
    #     for root_path in root_paths:
    #         for name1 in tqdm(os.listdir(root_path)):
    #             path1 = os.path.join(root_path, name1)
    #             for name2 in os.listdir(path1):
    #                 path2 = os.path.join(path1, name2)
    #                 name_set = set(os.listdir(path2))
    #                 name_list = list(set([os.path.splitext(x)[0] for x in os.listdir(path2)]))
    #                 for name3 in name_list:
    #                     if name3 + '.bmp' in name_set and name3 + '.json' in name_set:
    #                         path3 = os.path.join(path2, name3)
    #                         if test_json(path3 + '.json'):
    #                             f.write(path3 + '\n')

def gen_label():
    count = 0
    patient_label = {}
    label_name = {}
    df = pd.read_csv(label_path)
    df = df[['patient id', 'label']]
    for num, row in df.iterrows():
        if num % 50 == 0:
            print('[{}]'.format(num))
        if row[1] not in label_name:
            count += 1
            label_name[row[1]] = count
        patient_id = int(row[0][-5:])
        patient_label[patient_id] = label_name[row[1]]
    label_name = {v: k for k, v in label_name.items()}
    content = {
        'patient label': patient_label,
        'label name': label_name
    }
    with open('label.json', 'w') as f:
        content = json.dumps(content)
        f.write(content)
    print('[Finish]')

def data_yuanfa_zhuanyi():
    count = 0
    train_data = []  # (patient id, data path)
    test_data = []
    patient_name =[]
    str_1 = "原发"
    str_2 = "转移"
    str_3 = "骨髓瘤"
    str_4 = "淋巴瘤"
    str_5 = "轴位"
    str_6 = "矢状位"
    df = pd.read_csv(label_path)
    df = df[['patient id', 'label']]
    for num, row in df.iterrows():
        # if (str_3 in row[1]) or (str_4 in row[1]):
        if (str_3 in row[1]) :
            patient_name.append(row[0][9:21])
    patient_name_set = set(patient_name)
    
    root_paths = ['/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/李媛/MRI/',
                  '/home/fym/dataset/TOSHIBA_EXT//原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/刘剑芳/MRI/',
                  '/home/fym/dataset/TOSHIBA_EXT//原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/欧阳汉强/MRI/',
                  '/home/fym/dataset/tumer_2/转移瘤分位置/MRI2/李媛MRI2/',
                  '/home/fym/dataset/tumer_2/转移瘤分位置/MRI2/欧阳汉强MRI2/']
    # root_paths = ['/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/李媛/CT骨窗/',
    #               '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/李媛/CT软组织窗/',
    #               '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/刘剑芳/CT骨窗/',
    #               '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/刘剑芳/CT软组织窗/',
    #               '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/欧阳汉强/CT骨窗/',
    #               '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/欧阳汉强/CT软组织窗/',
    #               '/home/fym/dataset/tumer_2/转移瘤分位置/CT2/李媛CT2',
    #               '/home/fym/dataset/tumer_2/转移瘤分位置/CT2/欧阳汉强CT2',
    #               '/home/fym/dataset/data_xin/6转移瘤验证组bmp/刘剑芳/软组织窗新',
    #               '/home/fym/dataset/data_xin/6转移瘤验证组bmp/刘剑芳/骨窗新',
    #               '/home/fym/dataset/data_xin/6转移瘤验证组bmp/李媛/软组织窗新',
    #               '/home/fym/dataset/data_xin/6转移瘤验证组bmp/李媛/骨窗新',
    #               '/home/fym/dataset/data_xin/6转移瘤验证组bmp/欧阳汉强/软组织窗新',
    #               '/home/fym/dataset/data_xin/6转移瘤验证组bmp/欧阳汉强/骨窗新']
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
                                if str_1 in root_path:
                                    if name1[0:12] in patient_name_set:
                                        label_ID =1
                                        train_data.append((label_ID, path3))
                                elif str_2 in root_path:
                                    label_ID =2
                                    train_data.append((label_ID, path3))
    
    root_paths = ['/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/王春杰/MRI/',
                '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/袁源/MRI/',
                '/home/fym/dataset/tumer_2/转移瘤分位置/MRI2/王春杰MRI2/']
    # root_paths = ['/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/王春杰/CT骨窗/',
    #             '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/王春杰/CT软组织窗/',
    #             '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/袁源/CT骨窗/',
    #             '/home/fym/dataset/TOSHIBA_EXT/原发瘤bmp/脊柱肿瘤-原发瘤/标注后的bmp/袁源/CT软组织窗/',
    #             '/home/fym/dataset/tumer_2/转移瘤分位置/CT2/王春杰CT2',
    #             '/home/fym/dataset/data_xin/6转移瘤验证组bmp/王春杰/软组织窗新',
    #             '/home/fym/dataset/data_xin/6转移瘤验证组bmp/王春杰/骨窗新',
    #             '/home/fym/dataset/data_xin/6转移瘤验证组bmp/袁源/软组织窗新',
    #             '/home/fym/dataset/data_xin/6转移瘤验证组bmp/袁源/骨窗新']
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
                            if str_1 in root_path:
                                label_ID =1
                                if name1[0:12] in patient_name_set:
                                    if str_6 in path3:
                                        test_data.append((label_ID, path3))
                            elif str_2 in root_path:
                                label_ID =2
                                if str_6 in path3:
                                    test_data.append((label_ID, path3))
    train_data = filter_data(train_data)
    test_data = filter_data(test_data)
    content = {
        'train_data': train_data,
        'test_data': test_data
    }
    with open('./data_mr_gusuiliu_zhuanyiliu_S.json', 'w') as f:
        content = json.dumps(content)
        f.write(content)

def init_dataset_paramaters():
    dataset = MedicalDataset('train', 'data.json', 'label.json')
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
    for i in tqdm(range(len(dataset))):
        img, target = dataset[i]
        assert isinstance(img, torch.Tensor)
        img = img.double().reshape((3, -1))
        img = (img - mean[:, None]) * (img - mean[:, None])
        std += img.sum(dim=-1)
    std = torch.sqrt(std / num)
    mean = list(mean.tolist())
    std = list(std.tolist())
    return mean, std


def init_data_distribution():
    label_name = read_json('label.json')['label name']
    dataset = MedicalDataset('train', 'data.json', 'label.json')
    num = np.zeros((18,), dtype=np.int).tolist()
    for item in tqdm(range(len(dataset))):
        img, target = dataset[item]
        lables = target['labels'].tolist()
        for x in lables:
            num[x - 1] += 1
    order = list(range(18))
    order.sort(key=lambda x: num[x], reverse=True)
    order_dict = {}
    db = {}
    for i in range(len(order)):
        order_dict[order[i] + 1] = i + 1
        db[i + 1] = num[order[i]]
    with open('order.json', 'w') as f:
        content = json.dumps(order_dict)
        f.write(content)
    with open('distribution.json', 'w') as f:
        content = json.dumps(db)
        f.write(content)


if __name__ == "__main__":
    # gen_label()
    # gen_dataset_path()
    # init_data_path()
    # data_yuanfa_zhuanyi()
    # init_data_distribution()
    # mean, std = init_dataset_paramaters()
    # print('mean:', mean)
    # print('std:', std)
    pass
