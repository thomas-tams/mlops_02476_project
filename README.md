# mlops_project_tcs

Casper Harreby - s204558
Thomas Tams - s204540
Sven Finderup - 

MLOps project from the DTU course *02476 - Machine Learning Operations*

## Project description
Brain MRI image analysis is an extensive procedure requiring many advanced tools and expert knowledge to properly use the images for diagnosis. The time from the scanning of has occurred to a diagnosis is given to the patient can take several days. To speed up and support this process, deep learning models provide many tools. Convolutional neural networks have shown promising results in solving complex imaging tasks and much research is centered around applying these tools in medical imaging.

The overall goal of the project is to build a CNN image classifier. The model should be able to input an MRI image of a brain scan and detect if there is a tumor present or not. The goal is to achieve relatively high accuracy using powerful pretrained network in a reproducible setting.

The framework used in the project will be a PyTorch implementation in VS code using GitHub collaboration. The projects aims to both have a local setup but also docker reproducibility.

The data for this project consists of 98 brain scan images with no tumors and 279 images with tumors. The data exists as part of a kaggle challenge 

For this task, a relevant convolutional NN architecture will be used such as VGG-16 or ResNet.


## Init project using pip

Create a python>=3.9 environment with pip>=24.2

Example using conda:
```
$ conda create -n py39_mlops python=3.9
$ conda activate py39_mlops
```

To install the python package and depencies run
```
$ pip install .
```




## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
