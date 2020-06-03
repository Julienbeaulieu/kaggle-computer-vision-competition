Bengali.AI
==============================

This is the combined effort to build a pipeline in PyTorch to attempt Kaggle competition for Bengali.AI

In this competition, we are given the images of a handwritten Bengali grapheme and are challenged to separately classify three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics. 

Project writeup: https://julienbeaulieu.github.io/2020/03/16/building-a-flexible-configuration-system-for-deep-learning-models/

Implementation Highlights
------------
YAML config files, Mixup & albumentation library augmentations, label smoothing, OHEM, label weights for class imbalance, progressive resizing, OneCycleLR, Tensorboard. 
Models: MobileNet_V2, DenseNet121, SE_ResNeXT50


Project Organization
------------
Based on Facebook Research Detectron2 project and Cookie Cutter data science


    ├── README.md          
    ├── data
    │   ├── external        <- Data from third party sources.
    │   ├── interim         <- Intermediate data that has been transformed.
    │   ├── processed       <- The final, canonical data sets for modeling.
    │   └── raw             <- The original, immutable data dump.
    │
    ├── notebooks           <- Jupyter notebooks for exploration and communication
    │
    ├── experiments         <- Trained models, predictions, experiment configs, Tensorboard logs, backups
    │   ├─ exp01
    │   │  ├── model_backups
    │   │  ├── results
    │   │  └── tensorboard logs
    │   │
    │   ├─ exp02
    │   ...
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment,
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── config
        │   ├── config.py  <- The default configs for the model. 
        │   └── densenet121
        │       ├── exp01_config.yaml <- Configs for a specific experiment. Overwrites default configs
        │       └── exp02_config.yaml
        │       
        ├── data                  
        │   ├── make_dataset.py  <- Script to generate data
        │   ├── bengali_data.py  <- Custom Pytorch Dataset & Collator class; build_dataloader function
        │   └── preprocessing.py <- Custom data augmentation class
        │
        ├── modeling         <- Scripts to create the model's architecture             
        │   ├── backbone     <- Model's backbone architecture
        │   │   └── densenet121.py
        │   │
        │   ├── layers       <- Custum layers specific to your project
        │   │   └── linear.py
        │   │
        │   ├── meta_arch    <- Scripts to combine and build backbone + layers + head
        │   │   ├── baseline.py
        │   │   └── build.py
        │   │
        │   ├── head         <- Build the head of the model - in our case a classifier
        │   │   ├── build.py
        │   │   └── simple_head.py
        │   │
        │   └── solver       <- Scripts for building loss function, evaluation and optimizer
        │       ├── loss
        │       │   ├── build.py
        │       │   ├── softmax_cross_entropy.py
        │       │   └── label_smoothing_ce.py
        │       ├── evaluation.py
        │       └── optimizer.py 
        │ 
        ├── tools            <- Training loop and custom tools for the project
        │   ├── train.py
        │   └── registry.py 
        │ 
        └── visualization  <- Scripts to create exploratory and results oriented visualizations or to store reports
               └── visualize.py


