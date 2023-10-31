# Since the dataset is confidential, the actual dataset files have not been uploaded. 
# The "train_json_path," "eval_json_path," and "test_json_path" in the code represent the datasets for training, validation, and testing, respectively. 
# These files contain the paths to each image and their corresponding annotations. 
# "f_img" represents the read image, and "zlevel" and "level" are the label corresponding to intervertebral disc degeneration and spinal canal stenosis for that image, respectively.
import torch
import cv2
import numpy as np
import json
from matplotlib import patches as patches
from torch.utils.data import Dataset
from torch.autograd import Variable

train_json_path = ''
eval_json_path = ''
test_json_path = ''


"zlevel和level一起返回"
class MedicalDataset(Dataset):

    def __init__(self, mode, transform = None):

        assert mode == 'train' or mode == 'eval' or mode == 'test'
        self.mode = mode
        self.transform = transform
        data_path = {
                    'train_data':train_json_path,
                    'eval_data':eval_json_path,
                    'test_data': test_json_path
                     }
        if self.mode == 'train':
            self.path = data_path['train_data']
        elif self.mode == 'eval':
            self.path = data_path['eval_data']
        else:
            self.path = data_path['test_data']

        print(self.path)
       
        self.data = self.read_json(self.mode,self.path)

    @staticmethod
    def read_json(mode,save_path, encoding='utf8'):   
        jsondata_pos_path = []
        all_data = []
        with open(save_path, 'r', encoding=encoding) as f:
            content = f.read()
            content = json.loads(content)
            for key in content:
                for i in range(len(content[key])): 
                    data = {'key':content[key][i]['key'], 'img_path':content[key][i]['img_path']}
                    if content[key][i]['key'] == True and content[key][i]['img_path'] not in jsondata_pos_path :
                        jsondata_pos_path.append(content[key][i]['img_path'])
                        bmp_path = content[key][i]['img_path']
                        
                        #img
                        img = cv2.imread(bmp_path)
                        if img is None:
                            print("img is none\n")
                        if img.shape[0] == 384:
                            img = np.pad(img, pad_width = ((64,64),(0,0),(0, 0)), mode = 'constant',  constant_values = (0))
                        img  = img[130:280,171:341] #cut
                        img = cv2.resize(img, (224, 224))
                        img = np.float32(img)
                        img = np.ascontiguousarray(img[..., ::-1])
                        img = img.transpose(2, 0, 1)
                        # Convert to float tensor
                        img = torch.from_numpy(img)
                        # Convert to Pytorch variable
                        f_img = Variable(img, requires_grad=False)

                        zlevel = 0
                        if content[key][i]['zlevel'] in[0,1]:
                            zlevel = 0
                        elif content[key][i]['zlevel'] in[2,3]:
                            zlevel = 1

                        level = 0
                        if content[key][i]['level'] in[0,1]:
                            level = 0
                        elif content[key][i]['level'] in[2,3]:
                            level = 1
                       
                       
                        label = torch.tensor([float(zlabel),float(level)])
                        f_data = [f_img, label,content[key][i]['pos'],bmp_path]

                        all_data.append(f_data)

        return all_data
        

  
    def __len__(self):        
        return len(self.data) 
    
    def __getitem__(self, item):
        return self.data[item]



if __name__ == '__main__':
    pass
    # MedicalDataset('train')
    # MedicalDataset('eval')
    # MedicalDataset('test')

