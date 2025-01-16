# mlops_project_tcs

Casper Harreby - s204558
Thomas Tams - s204540
Sven Finderup -

MLOps project from the DTU course *02476 - Machine Learning Operations*

## Project description
Brain MRI image analysis plays a critical role in diagnosing conditions such as brain tumors. This process typically requires advanced tools and expert interpretation, often taking days to provide a diagnosis. To accelerate and support this workflow, deep learning models offer effective solutions. Among these, convolutional neural networks (CNNs) have demonstrated exceptional performance in complex imaging tasks, making them a valuable tool in medical imaging.

This project aims to develop a CNN-based classifier capable of detecting the presence of tumors in brain MRI scans. The classifier will take an MRI image as input and output whether a tumor is present, targeting high accuracy through the use of pretrained network architectures in a reproducible and collaborative framework.

The implementation will utilize PyTorch and be developed in VS Code with version control and collaboration facilitated through GitHub. The project emphasizes reproducibility by incorporating a local development setup alongside a Dockerized environment.

The dataset for this task is publicly available from a Kaggle challenge and consists of 98 images without tumors and 279 images with tumors. These images will be preprocessed and used to train and evaluate the model.

For the model architecture, state-of-the-art CNNs such as VGG-16 and ResNet will be explored. These networks, pretrained on large image datasets, provide robust feature extraction capabilities and can be fine-tuned for the specific task of tumor classification. By leveraging these powerful architectures, the project aims to produce a reliable and efficient tool for medical imaging analysis, contributing to faster and more accurate diagnoses.


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



## Init project using docker

Requires invoke>=2.2.0

To build docker images run
```
$ invoke docker-build
```

### Setup with with gpu support

Requires [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

To build all images including gpu supported run
```
$ invoke docker-build-gpu
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
