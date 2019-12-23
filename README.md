# Machine Learning Engineer Nanodegree

## Capstone Project: Automatic Tag Generation for StackOverflow Questions
### By: Pablo Campos Viana

## Requirements

This project requires  **Python 3.6**  and the kernel **conda_pytorch_p36** in Amazon SageMaker to execute the ``Project.ipynb`` file. Additionally, the following Python libraries are required:

-  [pandas 0.25](https://pandas.pydata.org/pandas-docs/stable/)
- [wordcloud](https://pypi.org/project/wordcloud/)
-  [Kaggle API](https://pypi.org/project/kaggle/)
- [fastText](https://pypi.org/project/fasttext/)
-  [torchtext](https://pypi.org/project/torchtext/)

## Folders and files

The following folders and files are provided:

- ``proposal.pdf``: The capstone proposal for this project.
- ``report.pdf``: The final report for this project.
- ``Project.ipynb``: The development Notebook used for this project and required to reproduce the solutions and results.
- *stacksample*: It will be created after executing the notebook ``Project.ipynb``. It will contain the  ``Questions.csv`` and ``Tags.csv`` tables downloaded via the Kaggle API.
- *forBlazingText*: It will be created after executing the notebook ``Project.ipynb``. It will contain the ``train``,``valid``, and ``test`` text files required for both fastText and BlazingText models.
- *forPyTorch*: It will be created after executing the notebook ``Project.ipynb``. It will contain the ``train.csv``,``valid.csv``, and ``test.csv`` csv files required for the PyTorch model.
-  *source*: Contains the custom code required for training and serving the PyTorch model.
- *website*: Contains the HTML file required for deploying the web app that uses the deployed PyTorch model.

Additionally, a `kaggle.json` file containing your Kaggle API credentials (more info [here](https://github.com/Kaggle/kaggle-api)) is required to be located in the same directory of these folders so it can be used to download the [stacksample](https://www.kaggle.com/stackoverflow/stacksample) data.
