# Medi_diagnosis
# COVID-19 and Pneumonia Detection using CNN

This project uses Convolutional Neural Networks (CNNs) to detect Pneumonia and COVID-19 from medical images (X-Rays and CT Scans). It was developed as part of a 6-week internship at the RAPID (Research in AI for development of Interdisciplinary Sciences) center at PES University.

## Table of Contents
- [Overview](#overview)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Challenges](#challenges)
- [Lessons Learned](#lessons-learned)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgements)

## Overview

The internship was conducted at the research center of the Department of AIML in the Ring road campus of PES University. RAPID (Research in AI for development of Interdisciplinary Sciences) aims to leverage the growing opportunities of artificial intelligence and machine learning towards developing other fields such as medicine, finance, infrastructure, etc.

This project focuses on developing CNN-based models for the detection of Pneumonia and COVID-19 in medical images, specifically X-Rays and CT Scans. We developed three models and integrated them into a user-friendly Streamlit application.

## Models

1. **Pneumonia Detection**: Analyzes chest X-rays to detect the presence of Pneumonia.
2. **COVID-19 X-Ray Detection**: Examines chest X-rays to identify potential COVID-19 cases.
3. **COVID-19 CT Scan Detection**: Analyzes CT scans to detect the presence of COVID-19.

All models are based on the ResNet-18 architecture, which was chosen for its depth, efficiency, and robustness in handling medical imaging tasks. The reasons for choosing ResNet-18 include:

1. **Depth**: The depth allows it to learn complex patterns in the data, crucial for identifying slight differences in medical images.
2. **Efficiency**: Despite its depth, ResNet-18 is relatively efficient in terms of computational resources.
3. **Robustness**: Skip connections contribute to the model's robustness and help it generalize better to unseen data.

## Usage

To run the Streamlit app:

1. Ensure you're in the project directory and your virtual environment is activated.
2. Run the following command:
   ```bash
   streamlit run app.py

## Results

Our models achieved the following performance metrics:

### Pneumonia Detection Model
- Accuracy: 83.606%
- Precision: 61.07%
- Recall: 75.206%
- F1 Score: 67.407%

### COVID-19 X-Ray Model
- Accuracy: 92.23%
- Precision: 93.52%
- Recall: 96.14%
- F1 Score: 94.81%

### COVID-19 CT Model
- Accuracy: 92.55%
- Precision: 74.72%
- Recall: 96.56%
- F1 Score: 85.10%

These results demonstrate excellent performance across all three models. When tested with random medical images from various datasets, the models consistently provided accurate predictions for the detection of the respective diseases.



