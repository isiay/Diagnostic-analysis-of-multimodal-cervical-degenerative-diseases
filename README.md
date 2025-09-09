This repository is the official implementation of the paper: "Automated Diagnostic of Cervical Spondylosis on Multimodal Medical Images with a Multi-task Deep Learning Model".

# X-ray model

To train the X-ray model, run the following command:

```
cd ~/mmpose

./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_512x512.py 4 --no-validate --autoscale-lr

# Optional: Resume training from a specific checkpoint
./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_512x512.py 4 --no-validate --autoscale-lr --resume-from work_dirs/baseline1-higherhrnet_w32_coco_512x512/latest.pth
```

To evaluate the trained model：

```
cd ~/mmpose

CONFIG_FILE="configs/inference_only.py"
CHECKPOINT_FILE="work_dirs/baseline2-softmax3Loss_300epoch_higherhrnet_w32_coco_512x512/latest.pth"
RESULT_DIR="test/"
RESULT_FILE=$RESULT_DIR"result.json"

python tools/inference.py $CONFIG_FILE $CHECKPOINT_FILE --out $RESULT_FILE 
```

# MRI model

To train the MRI model, run the following command:

```
cd ~/MRI_code_detection

python train.py
```

To evaluate the trained model:
```
cd ~/MRI_code_detection

python visual_paint_verify.py
```

# Dataset

Since the dataset contains confidential information, the actual data files are not included in this repository. However, the dataset will be made publicly available upon the publication of the accompanying paper.


