# AI_Fianl_Project

This project aims to solve the problem proposed by [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection/submissions) on Kaggle.

### Data Preparation
The files should be placed as follows:
```
repo
  +- train
  |  +- ***.tif
  |  +- ***.tif
  |  +- ...
  |
  +- test
  |  +- ***.tif
  |  +- ***.tif
  |  +- ...
  |
  +- train_labels.csv
  +- train.py
  +- util.py
  +- infer.py
  +- ResNext.py
  |  ...
```

### Training
To reproduce the training process, simply run train.py. The model is trained with 4 GPUs, so you may need to change the GPU setting and batch size in the code at the begining.

### Inference
Simply run infer.py, and a csv file of predictions should be created.
