import argparse
import torch
import torch.utils.data
import config
import os
from detection import generate_model
# from utils.engine import evaluate
from dataset.medical_dataset_img_4 import MedicalDataset
from utils import utils
from tqdm import tqdm
import cv2
import json
import cv2
import numpy as np
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 



path = '/home/myy/jingzhui/MRI/MRI_data_prepare/show_pic'

def resize_ann(img,size):
    cv2.imwrite(path + '/'+ 'ann_resize_yuan.jpg', img)
    original_img = cv2.imread(path + '/'+ 'ann_resize_yuan.jpg', cv2.IMREAD_GRAYSCALE)
    print(original_img.shape)
    resize_img = cv2.resize(original_img, size, cv2.INTER_NEAREST)
    print(resize_img.shape)
    cv2.imwrite(path + '/'+ 'ann_resize.jpg', resize_img)
    return resize_img

def return_ann(img,size):
    resize_to_org_img = cv2.resize(img, (size[1],size[0]), cv2.INTER_LANCZOS4)
    print(resize_to_org_img.shape)
    cv2.imwrite(path + '/'+ 'ann_return.jpg', resize_to_org_img)
    return resize_to_org_img

def heat_map(list_features):
    
    # list_features = list_features.cpu().detach().numpy() 
    # print(list_features.keys())
    # print(type(list_features))
    # heatmap = list_features.cpu().detach().numpy()
    # for heatmaps in list_features:
    # print(type(heatmaps))
    # heatmap = np.array(list_features)
    heatmap = list_features
    # print(heatmap)
    # for heatmap in heatmaps:
    # print(heatmap.shape)
    v_min = heatmap.min()
    v_max = heatmap.max()
    # print(v_min,v_max)
    # print(v_max - v_min)
    heatmap = (heatmap - v_min) / max((v_max - v_min), 1e-10)
    print(heatmap.max(), heatmap.min())
    heatmap = cv2.resize(heatmap,(512,512)) * 255
    heatmap = resize_ann(heatmap,(64,64))
    heatmap = return_ann(heatmap,(512,512))

    heatmap = heatmap.astype(np.uint8)
    # cv2.imshow("heatmap1",heatmap)
    cv2.imwrite("/home/myy/jingzhui/MRI/MRI_data_prepare/show_pic/heatmap1.png", heatmap)
    # cv2.waitKey()
    heatmap2 = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # heatmap2 = cv2.cvtColor(heatmap,cv2.COLOR_GRAY2BGR)
    # cv2.imshow("heatmap2", heatmap2)
    cv2.imwrite("/home/myy/jingzhui/MRI/MRI_data_prepare/show_pic/heatmap2.png", heatmap2)
    # cv2.waitKey()
    # superimposed_img = heatmap2 * 0.4 + img * 0.6
    # superimposed_img = np.clip(superimposed_img,0,255).astype(np.uint8)
    # # cv2.imshow("superimposed_img", superimposed_img)
    # cv2.imwrite("/home/myy/jingzhui/MRI/MRI_data_prepare/show_pic/superimposed_img.png", superimposed_img)
    # cv2.waitKey()

def collate_fn(batch):
    return tuple(zip(*batch))


def main(deformable=False, model_path='/home/myy/jingzhui/MRI/MRI_pth_detection/classfier_4_kong2_b2.pth'):        
    # print(model_path)
    IOU_list = []
    num2 = 0
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    # evaluate(model, test_data_loader, device, None)
    num_classes = 5  # classes
    mode = 'show'
    gt = 2  #1 level 2 levelx 3 pred 画不画gt框？
    inx = 1
    patient = set()
    print(inx)
    test_dataset = MedicalDataset(mode, config.label_path)
    # test_dataset = MedicalDataset('train', config.data_path, config.label_path)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                   num_workers=2, collate_fn=collate_fn)
    model = generate_model.faster_rcnn(num_classes=num_classes, deformable=deformable)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    # print(torch.load(model_path).keys())
    model.eval()
    num1 = len(test_dataset)
    # c = 0
    # c1 = 0  #01，23
    # c2 = 0  #0，123
    # # s  = 0.5
    # t = 0
    # print(s)
    with torch.no_grad():
        # for idx, data in enumerate(test_dataset):
            # idx = 7
           # 可视化demo
        # for idx in range(0,10):
        for idx in range(len(test_dataset)):
            print(idx)
            # print(test_dataset.data[idx]['img_path'])   #输入的数据
            bmp_path = test_dataset.data[idx]['img_path']
            patient.add(str(bmp_path).split('/')[-2])
            if len(patient) > 30:
                break
            id = str(bmp_path).split('/')[-2]+'_'+ str(bmp_path).split('/')[-1].split('.')[0]
            print(id)
            # print(test_dataset[idx])        #dataset得到的target
            src_img = cv2.imread(bmp_path)
            annotation = test_dataset.data[idx]['mask'] # 标注框
            # 标注框画上
            a = annotation[0]
            b = annotation[1]
            area_gt = (b[0] - a[0])*(b[1] - a[1])
            # tar2 = int(target) 
            # color = [(0,255,102),(0,204,204),(0,102,204),(0,51,204)]
            color = (0,255,102)
            # color = 255 * tar2 / 4
            if gt == 1 or gt == 2:  # 高年资or低年资的意思
                cv2.rectangle(src_img, tuple(map(int, a)), tuple(map(int, b)),
                            (color), thickness=2)  # over

            # 只需要标注的情况
            if mode == 'show':
                    save_path = '/home/myy/jingzhui/MRI/MRI_vertify/ann_30_check/{}'.format(inx)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_path+'/' + id + '.jpg' , src_img)





                

        '''
        for gt in range(1,4):
            for idx in range(len(test_dataset)):
                # print(idx)
                # print(test_dataset.data[idx])   #输入的数据
                # print(test_dataset[idx])        #dataset得到的target
                start = time.time()
                img = test_dataset[idx][0]
                if gt == 2:
                    target = test_dataset.data[idx]['levelx']
                else:
                    target = test_dataset.data[idx]['level']
                # print(target)
                bmp_path = test_dataset.data[idx]['img_path']
                # bmp_path2 = bmp_path[1] + '.bmp'
                # patient = str(bmp_path2).split('/')[-3]
                # print(patient)
                # if (str_2 in bmp_path2) :
                # if str_1 in bmp_path2:
                #     name1 = '轴位'
                #     name2 = 'Axial'
                # elif str_2 in bmp_path2:
                #     name1 = '矢状位'
                #     if (str_3 in bmp_path2) and (str_5 not in bmp_path2):
                #         name2 = 'T1'
                #     elif (str_3 in bmp_path2) and (str_5 in bmp_path2):
                #         name2 = 'T1-增强'
                #     elif (str_4 in bmp_path2) and (str_5 not in bmp_path2):
                #         name2 = 'T2'
                #     elif (str_4 in bmp_path2) and (str_5 in bmp_path2):
                #         name2 = 'FS-T2'
                # json_path = test_dataset.data[idx]  # 答案框绘制
                # json_path = json_path[1] + '.json'
                src_img = cv2.imread(bmp_path)
                
                img_tensor = img
                detections = model([img_tensor.to(device)])

                if gt == 1:
                    save_path = '/home/myy/jingzhui/MRI/MRI_pth_detection/show/ppt_1_v3'
                elif gt == 2:
                    save_path = '/home/myy/jingzhui/MRI/MRI_pth_detection/show/ppt_1_v3_di'
                else:
                    save_path = '/home/myy/jingzhui/MRI/MRI_pth_detection/show/ppt_1_v3_pre'
                if os.path.exists(save_path+'/' + bmp_path.split('/')[-1]):
                    src_path = save_path+'/' + bmp_path.split('/')[-1]
                    src_img = cv2.imread(src_path)
                else:
                    if gt == 'yes':
                        save_path = '/home/myy/jingzhui/MRI/MRI_pth_detection/show/ppt_2_v3'
                    elif gt == 'y':
                        save_path = '/home/myy/jingzhui/MRI/MRI_pth_detection/show/ppt_2_v3_di'
                    else:
                        save_path = '/home/myy/jingzhui/MRI/MRI_pth_detection/show/ppt_2_v3_pre'
                    if os.path.exists(save_path+'/' + bmp_path.split('/')[-1]):
                        src_path = save_path+'/' + bmp_path.split('/')[-1]
                        src_img = cv2.imread(src_path)

                    # cv2.imshow('IMG', src_img)
                    # print(src_img)
                annotation = test_dataset.data[idx]['mask']
                # annotation = json.loads(open(json_path, encoding='gbk').read())['shapes']
                # print(annotation)
                a = annotation[0]
                b = annotation[1]
                area_gt = (b[0] - a[0])*(b[1] - a[1])
                tar2 = int(target) 
                color = [(0,255,102),(0,204,204),(0,102,204),(0,51,204)]
                # color = 255 * tar2 / 4
                if gt == 1 or gt == 2:
                    cv2.rectangle(src_img, tuple(map(int, a)), tuple(map(int, b)),
                                (color[int(tar2)]), thickness=2)  # over
                # print(detections)
                boxes = detections[0]['boxes']
                labels = detections[0]['labels']
                # print(labels)
                scores = detections[0]['scores']

                
                # if target == 0:
                #     tar2 = 0
                #     # print(target)
                # else:
                #     tar2 = 1
                    # for num2 in range(boxes.shape[0]):  # 预测框绘制
                    # print(scores)
                
                # print(len(boxes))
                if len(boxes) == 0:
                    iou = 0
                    IOU_list.append(iou)
                    num2 += 1
                    continue 

                box = boxes[0]
                score = scores[0]
                # Max = 0
                # print(scores, labels)
                # for i,data in enumerate(scores):
                #     if data > Max:
                #         Max = data
                #         box = i
                # print(list(labels.cpu().numpy()))
                pred = int(list(labels.cpu().numpy())[0]) - 1
                
                # if score < s:
                #     score = 1 - score
                #     pred = 1 - pred

                box = list(box.cpu().numpy())
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                area_pre = (x2-x1)*(y2-y1)
                xx1 = max(a[0], x1)  # 得到左上顶点的横坐标
                yy1 = max(a[1], y1)  # 得到左上顶点的纵坐标
                xx2 = min(b[0], x2)  # 得到右下顶点的横坐标
                yy2 = min(b[1], y2)  # 得到右下顶点的纵坐标
                w = xx2 - xx1
                h = yy2 - yy1
                area = w * h  # G∩P的面积
                            # print("G∩P的面积是：{}".format(area))
                if xx1 >= xx2 or yy1 >= yy2:
                    iou = 0
                else:
                    iou = area / (area_gt + area_pre - area)
                # rec2 = rect([a[0], a[1], b[0], b[1]])
                # print(rec1, rec2)
                # iou = compute_iou(rec1, rec2)
                IOU_list.append(iou)
                print(src_img.shape)
                # color = 255 * pred / 4
                
                if gt == 3:
                    cv2.rectangle(src_img,  tuple(map(int,(x1, y1))), tuple(map(int,(x2, y2))), (color[int(pred)]), thickness=2)
            
                if (len(scores) != 0):
                    num2 = 0
                    i = 0
                    # t = 0
                    while(num2 < 3 and i < len(scores)):
                        if scores[i] >= 0.5:
                            x1, y1, x2, y2 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                            area_pre = (x2-x1)*(y2-y1)
                            xx1 = max(a[0], x1)  # 得到左上顶点的横坐标
                            yy1 = max(a[1], y1)  # 得到左上顶点的纵坐标
                            xx2 = min(b[0], x2)  # 得到右下顶点的横坐标
                            yy2 = min(b[1], y2)  # 得到右下顶点的纵坐标
                            w = xx2 - xx1
                            h = yy2 - yy1
                            area = w * h  # G∩P的面积
                            # print("G∩P的面积是：{}".format(area))
                            IoU = area / (area_gt + area_pre - area)
                            if 0.5<IoU <=1:
                                IoU = IoU.cpu()
                        #     # print("IoU是：{}".format(IoU))
                                IOU_list.append(IoU)
                                num2 += 1
                                cv2.rectangle(src_img,  tuple(map(int,(x1, y1))), tuple(map(int,(x2, y2))), (0, 255, 0), thickness=2)
                            # mIOU = np.mean(IOU_list)
                            # print("平均IOU：{}".format(mIOU))
                            
                        i += 1
                # print(len(src_img))
                if mode == 'show':
                    # pix = []
                    # s1 = []
                    # boxes = boxes.cpu().numpy()
                    # l = np.zeros((512,512),dtype='float32')
                    # for i in range(len(boxes)):
                    #     x = boxes[i][0] + boxes[i][2]
                    #     x = int(x/2)
                    #     y = boxes[i][1] + boxes[i][3]
                    #     y = int(y/2)
                    #     s = scores[i] 
                    #     l[x][y] = s
                        # len1 = ((int(boxes[i][3]) - int(boxes[i][1])) ^ 2 + (int(boxes[i][2]) - int(boxes[i][0])) ^ 2 ) / 2
                        # # print(len1)
                        # for k in range(512):
                        #     for j in range(512):
                        #         # print(int(boxes[i][0]),int(boxes[i][2]),int(boxes[i][1]),int(boxes[i][3]))
                            
                        #         if k < int(boxes[i][0]) or k > int(boxes[i][2]) or j < int(boxes[i][1]) or j > int(boxes[i][3]):
                        #             continue
                        #         ss = ((k - x) ^ 2 + (j - y) ^ 2 ) / len1
                        #         if ss > 1:
                        #             continue
                        #         # print(ss)
                        #         l[k][j] += ss * s
                        #         # print(l[k][j])
                        # pix.append([x,y])
                        # s1.append(s)
                        # print(l)
                    # heat_map(l)
                    # print(boxes)
                    # print(scores)
                    # print(pix)
                    # print(s1)
                    # pass

                    if gt == 1:
                        save_path = '/home/myy/jingzhui/MRI/MRI_pth_detection/show/ppt_{}_v3'.format(inx)
                    elif gt == 2:
                        save_path = '/home/myy/jingzhui/MRI/MRI_pth_detection/show/ppt_{}_v3_di'.format(inx)
                    else:
                        save_path = '/home/myy/jingzhui/MRI/MRI_pth_detection/show/ppt_{}_v3_pre'.format(inx)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                        
                    # cv2.imwrite(save_path+'/'+str(idx)+'_'+ str(tar2)+ '_'+ str(pred) +'.jpg', src_img)
                    print(save_path)
                    cv2.imwrite(save_path+'/' + bmp_path.split('/')[-1] , src_img)

                # print(tar2,pred)
                if  tar2 == pred: 
                    c += 1

                if tar2 < 2 and pred < 2 or tar2 > 1 and pred > 1:  
                    c1 += 1

                if tar2 == 0  and pred == 0 or tar2 > 0 and pred > 0:
                    c2 += 1
                t += time.time() - start             
                print(tar2, pred)     
        # print(num2, c, c1, c2, num1)
        # mIOU = np.mean(IOU_list)
        # acc = c/num1
        # acc1 = c1/num1
        # acc2 = c2/num1
        # print("平均IOU：{}".format(mIOU))
        # print("分类ACC：{}".format(acc))
        # print(acc1,acc2,t/num1)
        '''

if __name__ == '__main__':
    main()