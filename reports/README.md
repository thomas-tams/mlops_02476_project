# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [x] Load test your application (M24)
* [x] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

 Group 84

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

 s204558, s204281, s204540

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

 In our project, we have implemented a sweep configuration file for hyperparameter optimization using the third-party framework Weights and Biases. We used functionalities such as sweep configuration and hyperparameter tuning from the package to define and run a sweep efficiently in our project.

The sweep setup uses the Bayesian optimization method, with predefined metrics such as validation loss and a goal to minimize it. The YAML configuration file includes a list of hyperparameters and their respective ranges or values, such as the number of epochs and various batch sizes, which will be explored during the training process.

When the training starts, the sweep launches instances locally and runs training with different combinations of hyperparameters. These runs communicate with the Weights and Biases server, which adjusts the hyperparameters between runs based on the optimization method. Finally, the sweep produces graphical representations of the training configurations alongside the corresponding validation accuracy, allowing us to analyze the performance of the models under different setups.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

 We used a requirements.txt file to manage our dependencies. The list of dependencies has been continually updated during development. To get a complete copy of our development environment, one would have to run the following commands: <br>`git clone https://github.com/thomas-tams/mlops_02476_project.git`
<br>`cd mlops_02476_project`
<br>`conda create -n mlops_project_tcs python=3.9`
<br>`conda activate mlops_project_tcs`
<br>`pip install -e .["dev"]`
<br>`pre-commit install`

Another smart way to get a copy of the development environment is by using *invoke*. With *conda* and *invoke* a new team member may set up a development environment through these commands using invoke in the base conda environment:
<br>`git clone https://github.com/thomas-tams/mlops_02476_project.git`
<br>`cd mlops_02476_project`
<br>`invoke create-environment`
<br>`conda activate mlops_project_tcs`
<br>`pip install invoke # Again in the new conda env`
<br>`invoke dev-requirements`
<br>`pre-commit install`

If the new developer needs to work with docker and Google Cloud, then you would have to install *gcloud* for the operating system and write your project's wandb_api.txt as well as getting a Google Cloud service API key with bucket admin permissions which you have to write to a file called bucket_manager.json.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

 From the cookiecutter template we have filled *.github*, *configs*, *data*, *dockerfiles*, *docs*, *reports*, *src*, and *tests* folders:
- The *.github* folder keeps the workflow .yaml files.
- In the *data* folder, we have created a subfolder structure containing raw, intermediary and processed data.
- In *configs* is a .json file with class labels.
- *dockerfiles* has our base .dockerfile together with gcloud .dockerfiles for API and FastAPI frontend.
- In the *docs* folder, we have app documentation and the mkdocs.yaml to create documentation (which we did not get to).
- *reports* contains our project hand-in report and script for creating .html rendition.
- In the *src* folder we have the project source code in *mlops_project_tcs*.
- *test* contains unit tests.
- We deviate from the *models* subfolder structure, since we save the 1 best model from a training run as onnx in the hydra *outputs* folder.
- We also use the hydra *outputs* folder for storing all our relavant training run data, such as logs, experiment configuration and best performing models.

We have removed the *notebooks* folder as we did note use any jupyter notebooks.

We have added *.dvc* to accomodate data version control containing the necessary configurations.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

 In order to ensure code quality and format, we have made use of the *Ruff* Python linter. In our source code functions, we have improved the quality of our code by applying *typing* to make it easier to directly read the types of input variables and returns. Additionally, docstrings and comments have been added to the code to enable more readily understandable logic. For a project like ours where more developers are envolved, this is essential. For example, the docstrings give a quick overview of parameters, arguments and returns. Furthermore, with *Ruff* integrated into our GitHub workflows, we make sure to continually check and format the source code as it is updated and developed.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

 In total, we have implemented three test .py scripts:
- *test_data_processing.py* tests the data preprocessing pipeline and whether the data is available and has the correct format.
- *test_model.py* tests that the model produces the correct output format (number of output classes).
- *test_train.py* creates a dummy dataset and test whether the trainer, hydra config files, wanb logger and initialization works correctly to run a training of the model.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

 In our project, the total code coverage of the tests is 49%, encompassing the most important half of our source code. We acknowledge that this figure is significantly below 100%. However, as emphasized in the course, achieving 100% code coverage does not guarantee that the code is error-free. This is because the unit tests themselves might not adequately test the underlying functionality or edge cases. Consequently, high code coverage can create a false sense of confidence in the correctness of the code. While code coverage provides an estimate of how much of the code is being tested, it does not measure the quality or completeness of the tests. Therefore, developing robust and meaningful unit tests is equally important and should complement code coverage metrics.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

Our workflow made use of both branches and pull requests. We used branches to implement new features, without effecting the main codebase. This allowed us to work on multiple task at the same time, without running the risk of ruining our code. Each feature could thus be developed and tested in a vacuum, allowing us to run tests before merging with the main branch.
By using pull requests all team members had the ability to review suggested changes and merge them into the main branch. Having this step in the process enabled us to more easily discuss potential changes, before actually implementing them. This meant that it was easier to ensure that the code lived up to the projects standards.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

 The data used in our project consists of MRI brain scan images, which we refer to as "raw data." We have utilized DVC (Data Version Control) exclusively for managing our raw data. This has been implemented by using a publicly accessible bucket in Google Cloud. While this approach hasn't been strictly necessary for our project—since the data originates from a Kaggle challenge with a fixed dataset—we anticipated that it could be a valuable feature if the project's scope was to expand. In such a case, a more comprehensive dataset might become available, and DVC would enable us to track and manage different versions of the brain scan dataset effectively.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:


We have implemented continuous integration (CI) for various parts of our project to maintain code quality and streamline our development workflow. Using pre-commit hooks, we ensure that workflows perform specific checks both locally during commits and in GitHub Actions. These hooks catch common issues. As mentioned earlier, we also run unit tests on our code in GitHub Actions to ensure that any flaws in updated code, which may not be detected locally during development, are identified and addressed. This ensures that every update integrates without introducing regressions. Additionally, we utilize Ruff as a pre-commit tool for linting, which helps us maintain a clean and consistent codebase by enforcing style guidelines and catching potential bugs early in the development process.

As an example of our continuous integration setup, we invite you to review one of our GitHub Actions workflows. This workflow ensures that when branches are merged into the main branch or commits are pushed to the main branch, a series of automated processes are triggered. These processes include installing dependencies and executing pytest to validate the integrity of our code. In the post-build stage (which runs only if the matrix build completes successfully), an updated Dockerfile for the project is sent to Google Cloud, where an updated Docker image is built. This ensures that any changes made to the project on GitHub are reflected in the Docker image.

By integrating these practices into our workflow, we maintain high standards for code quality, ensure reliability across updates, and minimize manual intervention in deployment processes. This setup enhances the project's reproducibility and ensures a seamless transition from development to production environments. You can view the workflow file here: https://github.com/thomas-tams/mlops_02476_project/blob/main/.github/workflows/tests.yaml



## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

 To configure our experiments, we used a *Hydra* setup with config files using the argparsing capabilities in *Hydra*. A code example of how we would use this could be shown when running the training of the model:
<br>`python src/mlops_project_tcs/train.py experiment.hyperparameter.n_epochs=1 experiment.hyperparameter.optimizer.weight_decay=0.0001`

During training, we use Weights and Biases to sweep over different configurations.


### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

 As already eluded to above, we made use of config files to set up experiments with predefined hyperparameters. With the functionalities in Hydra, we ensure that relevant information about the specific experiment is logged in an *outputs* folder. As mentioned earlier in Q5, we also save the .onnx model file for a specific run in the same folder. During development, if one wishes to run an identical experiment as one from the past, it can be done by accessing the config.yaml file that is saved in the aforementioned *outputs* folder in the following manner:
<br>`python src/mlops_project_tcs/train.py --config-dir outputs/<date>/<time>/.hydra --config-name config`

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:


![artifact_registry1](our_figures/wandb_graphs1.png)
![artifact_registry1](our_figures/wandb_graphs2.png)

In our experiments, we used Weights & Biases (W&B) to track key metrics and hyperparameters to evaluate and improve our model's performance systematically.

As seen in the first image, we tracked metrics such as train_loss, val_loss, train_acc, and val_acc over the training steps. These metrics are crucial for understanding the model's learning dynamics:

Training Loss and Validation Loss: These measure the model's performance on the training and validation datasets, respectively. A decreasing training loss indicates that the model is learning from the data, while a stable or decreasing validation loss suggests good generalization.

Training Accuracy and Validation Accuracy: These track how well the model predicts on training and validation sets. A divergence between these accuracies can highlight overfitting or underfitting issues.

In the second image, we conducted a hyperparameter sweep to analyze the impact of parameters such as dropout_p, learning rate, weight decay, batch size, and the number of epochs on the validation loss. The parameter importance chart indicates the significance of each parameter in influencing validation loss. For instance, dropout rate (dropout_p) and learning rate were identified as the most impactful parameters, which guided us in fine-tuning the model. The parallel coordinates plot visualizes how different combinations of hyperparameters correspond to validation loss, helping us select the best-performing configuration.

By logging these metrics and hyperparameters, we gained insights into model optimization and avoided manual trial-and-error processes. This tracking process was essential for ensuring reproducibility and improving the model's robustness.



### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:



We have used Docker in different ways in our project. During development, we have constructed interactive container sessions from the mlopsdtu_local_train, where we used GPU support, as well as mounted the outputs/ and models/ directories:

`docker run --rm --gpus all -it -v $(pwd)/outputs:/app/outputs -v $(pwd)/models:/app/models mlopsdtu_local_train sh`

In this session we setup a wandb sweep:

`wandb sweep src/mlops_project_tcs/sweeps/sweep.yaml`

And ran the training agent for training.

The Google Cloud based models are pushed to Google Cloud Artifact Registry and runs on Google Cloud Run services and are available at [FastAPI backend](https://mlops-api-707742802258.europe-west1.run.app/docs) and [Streamlit frontend](https://mlops-frontend-707742802258.europe-west1.run.app)


To access our project dockerfiles, use the following link:
https://github.com/thomas-tams/mlops_02476_project/tree/main/dockerfiles



### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

 When running experiments, we naturally encountered a lot of bugs and errors. We have tried to implement try/raise statements to detect errors. Many of our source code scripts are built on *typer* which has convenient error-messages. Besides, we have relied heavily on the VS Code built-in debugger. A major source of support has come from ChatGPT and GitHub copilot. Although there are many learnings in getting to know error and traceback messages properly, we also realized during the development that it can be a time-consuming task to debug and therefore utilized the AI tools availabe. Finally, we have run some profiling runs a few times via the PyTorch Lightning module.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:


We used the following services on Google Cloud:

Engine: For testing docker images and run capabilities in a cloud environment (during development).

Bucket: For storing DVC data publicly as well as onnx models for evaluation/prediction, for version control, backup and public availability.

Artifact Registry: For storing and accessing docker images in the cloud.

Cloud Build: For building cloud docker images via GitHub actions calls in CI.

Cloud Run: For hosting FastAPI backend and Streamlit frontend in the cloud for publicly available predictions.

We played around with Vertex AI, however we never got it to work properly.



### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:


We also tried to setup a few different virtual machine instances during our attempt to get training running in the cloud.
We tried to setup a simple CPU General purpose E2 instance (e2-medium (2 vCPU, 1 core, 4GB memory)) with 10 GB storage using a custom container image with our project installed.
We also created a GPU N1 instance using NVIDIA T4 GPU instance (n1-standard-2 (1 vCPU, 7.5 GB memory)) with 100 GB storage using one of google PyTorch container images (pytorch-latest-gpu-v20241224: Google, Deep Learning VM for PyTorch 2.4 with CUDA 12.4, M127, Debian 11, Python 3.10, with PyTorch 2.4 and fast.ai preinstalled)

We planned to use VMs for training, however inbetween getting the VMs set up, getting permissions for DVC, getting WandB working in the cloud, and service accounts set up in Google Cloud within a limited timeframe for the project, we opted to train on our local computers instead. This was possible seeing as the training runs were still rather quick and did have large requirements for hardware, since our model and dataset were somewhat small, atleast compared to models such as LLMs or other Deep Learning architectures.



### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:


![GCP bucket](our_figures/project-bucket.png)


### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:


![artifact_registry1](our_figures/artifact_registry1.png)
![artifact_registry2](our_figures/artifact_registry2.png)


### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:


![artifact_registry1](our_figures/cloud_build1.png)


### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:


We managed with the engine, but opted to run training locally, since we kept iterating and changing the source code quite often during development and did not figure out how to integrate the the newest updated version of the source code automatically into the VMs. We did not manage to get training running with Vertex AI. Here we struggled with GPU quotas for Vertex AI, service accounts premissions and injecting Weights and Bias api keys into the Vertex AI build/runtime.

It would have been nice and powerful to get these services up and running, since we would be able to scale training as well as run continously in the the cloud without downtime or running locally "locking" our computer. However, as mentioned earlier the model and datasets were somewhat light-weight and training locally was doable, due to low system requirements and GPU acceleration locally with an NVIDIA GeForce RTX 4060 Laptop GPU.



## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:


We managed to setup a FastAPI backend which is able take in a Brain MRI image, run an .onnx model and output prediction probabilities for our binary class problem. Furthermore, the API is also able to return a visualization of what the preprocessed input to the model would look like (for fun and education), which ended up helping us immensely in understanding and fixing a few quirks about or data preprocessing.
We also added a Streamlit frontend, which functions as a user interface, talking to the FastAPI backend, for easily uploading pictures and getting the prediction response and the visualization of the preprocessed input.




### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:


First we got the FastAPI and Streamlit frontend working locally, by serving the onnx model build from training runs. After this we build up docker images to accomodate the FastAPI and Streamlit frontend respectively locally. These images were then changed to accomodate Google Cloud Run platform, using environment variables to serve the port (via $PORT) and using Google Cloud Secrets for supplying the service account credentials. We also mounted a bucket to the FastAPI service, containing .onnx model which we used for evaluation/prediction of user inputs.

Here is an example of how to call the Google Cloud API
```
curl -X 'POST' \
  'https://mlops-api-707742802258.europe-west1.run.app/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@<image_path>.jpg;type=image/jpeg'
```


### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:


We tried implementing unittesting for the FastAPI backend, however we never succeeded due to errors regarding mocking of global variables assigned in the @asynccontextmanager in the api.py script.

We would have implemented the unittest using pytest and added these to the github action tests for push/pull request to main branch.

We did load testing of the FastAPI both when running api locally and in the Google Cloud. This we did via Locust package using the command.

```
locust -f src/mlops_project_tcs/locust_load.py --host https://localhost:8080
```

```
locust -f src/mlops_project_tcs/locust_load.py --host https://mlops-api-707742802258.europe-west1.run.app
```

Results of the runs were as follows
- Local API Test: API stopped responding after around 2500 concurrent users.
- Google Cloud API Test: using a ramp of 50 and 5000 concurrent users we got to around 2000 users doing 211.4 RPS with 52% Failure before stopping the tests.



### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:


In this project, we did not have time to implement the monitoring functionality from S8. However, monitoring would potentially have been a very important part of a project like this, where we work with detection of MRI brain scans of cancer patients. Assuming that the application was deployed and able to run, MRI imaging is also a scientific field under development. Hence, the data might change format as time goes by, and this new data might drift away from the data distribution that the model was trained on. Monitoring could help identify such data drift early, ensuring the model's predictions remain accurate. Furthermore, overall system monitoring of the application would also have greatly benefitted the project, allowing us to follow the user requests and system logs. It could also help in tracking model performance metrics over time, identifying cases where the model begins to underperform or where errors might occur. Lastly, monitoring would enable better troubleshooting and improve reliability by flagging issues in real-time, which is critical in a sensitive application like cancer detection.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:


We ended up spending a total of 1.55 credits. We used very few credits because we performed the model training on our local machine instead of in the cloud. The dataset used for this project is very small, which also ended up contributing to the small amount of credits used. The service that cost the most was Artifact Registry, which accounted for $0.77. This service is used to store and manage Docker images for our application. The second most expensive service was Cloud Run, which cost a total of 0.31 credits. Other services, such as Compute Engine and Cloud Storage, also incurred minor costs.

Overall, working in the cloud was a valuable experience, though it was challenging and occasionally frustrating. However, the possibilities it offers for deployment are significant, especially once you become familiar with the tools and workflows.



### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

 We invite you to look at the answer in question 23 where we talk about our frontend implementation.

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

 The overall architecture of our project is illustrated in the figure:
![Overall architecture figure](our_figures/project-overview.png)

The starting point of our pipeline is the local machine, where all the development takes place. Before the code is pushed to GitHub, pre-commit hooks are run to ensure correct formatting and adherence to coding standards. Once pushed to GitHub, GitHub Actions are triggered, running the same checks as pre-commit (in case pre-commit wasn't installed). Additional tests are also executed to verify functionality before the newly constructed Dockerfile is pushed to Google Registry, where the Docker image is built.

When training models on the local machine, the models are logged to Weights and Biases (W&B) for experiment tracking and monitoring. Hydra is used for hyperparameter optimization to construct the optimal model. Once the best model is selected, it is uploaded to Google Cloud Storage for storage and accessibility.

The Docker image is then deployed to Google Cloud Run, which hosts the backend application using FastAPI. The frontend is built using Streamlit, where users can upload images. These images are sent to the backend for processing and predictions. The results are then returned to the Streamlit interface for the user to view.

This structure ensures smooth integration of development, deployment, and user interaction, while maintaining reproducibility, experiment tracking, and model optimization.

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:


We mostly struggle with 4 things during the project.

The cloud and Google Cloud; We struggled alot with the cloud. From understanding the Console interface to the gcloud CLI. Understanding the service accounts and persmissions, as well as understanding glcoud CLI and all the command and possibilities through this. We did however manage to get an overview of both tools in some regard and we are utilizing a handful of Google Cloud tools in our project!

Docker and Secrets; Setting up docker images with premissions for WandB, Google Cloud Services and DVC. Initially the biggest struggle was understanding the process of setting these up locally with environment variables and later we struggle with implementing these variables through secrets in GitHub and Google Cloud. However, we managed to use the secrets from in GitHub and Google Cloud respectively in order to both update models through GitHub into the Google Cloud Artifact Registry, as well as give access the Google Cloud Run Containers running FastAPI and Streamlit.

Training in the cloud; We tried using both Compute Engine VMs and Vertex AI for training. We never managed to get Vertex AI running due, since we did not understand how to inject premission to WandB API and setup GPU quotas for Vertex AI. We did however manage to get a training run going on the VMs, however we opted not to use this for training anyways, since we did not implement a CI way of updating the code running on the VMs. With the fast iterations and changes to the source code, we thus found it easier and more productive to train locally.

Getting proper model predictions; All the models we trained did not converge in a good way. Always either guessing/predicting one a 100% on one class or 100% on the other class. This we could not manage to overcome, however we believe that this could be a due to the limited size of the dataset, simply not containing enough information for the model to understand the problem.



### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

 Overall, all students contributed equally to the completion in this project.

On a more granular level, some where more involved with some processes that others. s204540 did a lot of the actual coding and commits to the GitHub repo while s204281 and s204558 where more involved with ideation and conceptually designing the project and undestanding implementation of the project to the cloud which was a heavy task.
Additionally, as part of our project development we have used generative AI for code development, conceptual understanding and some text generation.
