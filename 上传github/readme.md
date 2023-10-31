### dataset.py (to prepare the dataset for training and testing)

Since the dataset is confidential, the actual dataset files have not been uploaded. The "train_json_path," "eval_json_path," and "test_json_path" in the code represent the datasets for training, validation, and testing, respectively. These files contain the paths to each image and their corresponding annotations. "f_img" represents the read image, and "zlevel" and "level" are the label corresponding to intervertebral disc degeneration and spinal canal stenosis for that image, respectively.

### train.py(to train the model)

The code trains models for classifying intervertebral disc degeneration and spinal canal stenosis. However, in the end, the model is only used for classifying intervertebral disc degeneration, while another model is used for classifying spinal canal stenosis.

### se_resnet.py(the mode file)

### utils_cgw.py(to save training logs and other information)

### MR_disic_pth(to save the parameters of the trained model)



