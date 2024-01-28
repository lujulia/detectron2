# NLPD :

I-CHEN LU


## Installation
Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html).
To use cityscapes, prepare data follow the [tutorial](https://detectron2.readthedocs.io/tutorials/builtin_datasets.html#expected-dataset-structure-for-cityscapes).

## Training

To train a model with 2 GPUs run:
```bash
cd /path/to/detectron2/projects/NLPD
python train_net.py --config-file configs/Cityscapes-PanopticSegmentation/panoptic_lmffnet_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml --num-gpus 2
```

## Evaluation

Model evaluation can be done similarly:
```bash
cd /path/to/detectron2/projects/NLPD
python train_net.py --config-file configs/Cityscapes-PanopticSegmentation/panoptic_lmffnet_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```
