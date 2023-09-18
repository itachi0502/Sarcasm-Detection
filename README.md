# Sarcasm-Detection

## Table of Contents
- [Project Overview](#project-overview)
  - [Project Background](#project-background)
  - [Key Features](#key-features)
  - [Technology Stack](#technology-stack)
- [Environment Requirements](#Environment-Requirements)
  - [Python Version](#Python-Version)
  - [Dependencies](#Dependencies)
- [Quick Start](#quick-start)
  - [Configuration File](#configuration-file)
    - [How to Edit the Configuration File](#how-to-edit-the-configuration-file)
  - [Data Preparation](#data-preparation)
    - [Dataset Source](#dataset-source)
    - [Contextual Data](#contextual-data)
    - [Data Format](#data-format)
  - [Training the Model](#training-the-model)

## Project Overview

### Project Background

Sarcasm is a prevalent rhetorical device encountered in social media, news comments, or everyday conversations. However, it presents a considerable challenge for Natural Language Processing (NLP) tasks. This project aims to develop an efficient and accurate sarcasm detection model to understand textual content more accurately in various applications such as sentiment analysis and public opinion monitoring.

### Key Features

- **Sarcasm Detection**: Utilizes advanced deep learning algorithms to classify the sarcastic sentiment of the input text.
- **Contextual Understanding**: Through interaction with relevant external knowledge bases (like Wikipedia), the model can grasp the text's context more accurately.
- **Multilevel Analysis**: Beyond basic text analysis, the model employs advanced features like multi-head attention mechanisms and bi-directional LSTM to capture complex relationships within the text.

### Technology Stack

- **PyTorch**: Used for model building and training.
- **Flair**: Employed for text classification and natural language processing tasks.
- **Transformer**: Word embeddings based on the Roberta model.
- **AdamW**: Serves as the model's optimizer.
- **OneCycleLR**: Utilized for learning rate scheduling.

## Environment Requirements

### Python Version

This project is implemented using Python 3.7.

### Dependencies

- PyTorch 1.13.1

Additional package dependencies can be found in the `requirements.txt` file and installed using pip:

```bash
pip install -r requirements.txt
```

## Quick Start

### Configuration File

The configuration for the model, optimizer, and other settings can be found in the YAML file (`iron.yaml`). Make sure to go through the configuration to understand the different parameters.

#### How to Edit the Configuration File

To edit the configuration file, simply open `iron.yaml` in a text editor and modify the values according to your needs. For example, to change the learning rate:

```yaml
train:
  learning_rate: 2e-5  # Change this value
```
### Data Preparation

#### Dataset Source
The dataset used for this project originates from the SemEval 2018 competition. The original dataset can be downloaded from [here](https://drive.google.com/file/d/1TKpxIm5Z6OSwxZdRR0ACOY6Edcygp0QQ/view?usp=drive_link)
#### Contextual Data
In this project, contextual information is added from Wikipedia, New York Times, and BBC. The processed dataset with context is available [here](https://drive.google.com/file/d/1M_pOwL7UW_8lkmC_9b0QQoUbEK0ttdUC/view?usp=drive_link)

