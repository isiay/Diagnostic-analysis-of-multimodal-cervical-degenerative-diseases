"""dataset config"""
data_path = {'train_data': '/home/myy/jingzhui/MRI/MRI_data_prepare/MRI_data_0_1_full_detection_pos_index_train.json', 'test_data': '/home/myy/jingzhui/MRI/MRI_data_prepare/MRI_data_0_1_full_detection_pos_index_test.json'}
label_path = '/home/myy/jingzhui/MRI/MRI_code_detection/label.json'
# order_path = '/home/fym/code/MR2/order.json'

"""train config"""
num_classes = 2
num_classes_filter1 = 3
num_epochs = 100

base_lr = 1e-2
momentum = 0.5
weight_decay = 0.0005

lr_power = 0.9
warmup_epoch = 3

image_mean = [0.036] * 3
image_std = [0.095] * 3
