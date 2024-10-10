# Medi_diagnosis
#COVID-19 and Pneumonia Detection using CNN
This project uses Convolutional Neural Networks (CNNs) to detect Pneumonia and COVID-19 from medical images (X-Rays and CT Scans). It was developed as part of an internship at the RAPID (Research in AI for development of Interdisciplinary Sciences) center at PES University.
Table of Contents

Overview
Models
Installation
Usage
Results
Contributors
Acknowledgements

Overview
This project aims to leverage AI and ML techniques to assist in the detection of respiratory diseases, specifically Pneumonia and COVID-19, using medical imaging. We developed three separate models using the ResNet-18 architecture and integrated them into a user-friendly Streamlit application.
Models

Pneumonia Detection: Analyzes chest X-rays to detect the presence of Pneumonia.
COVID-19 X-Ray Detection: Examines chest X-rays to identify potential COVID-19 cases.
COVID-19 CT Scan Detection: Analyzes CT scans to detect the presence of COVID-19.

All models are based on the ResNet-18 architecture, which was chosen for its depth, efficiency, and robustness in handling medical imaging tasks.
Installation
To set up this project, follow these steps:

Clone the repository:
Copygit clone https://github.com/your-username/covid-pneumonia-detection.git
cd covid-pneumonia-detection

Create a virtual environment and activate it:
Copypython -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required dependencies:
Copypip install -r requirements.txt


Usage
To run the Streamlit app:

Ensure you're in the project directory and your virtual environment is activated.
Run the following command:
Copystreamlit run app.py

Open your web browser and go to the URL displayed in the terminal (usually http://localhost:8501).

Results
Our models achieved the following performance metrics:

Pneumonia Detection Model:

Accuracy: 83.606%
Precision: 61.07%
Recall: 75.206%
F1 Score: 67.407%


COVID-19 X-Ray Model:

Accuracy: 92.23%
Precision: 93.52%
Recall: 96.14%
F1 Score: 94.81%


COVID-19 CT Model:

Accuracy: 92.55%
Precision: 74.72%
Recall: 96.56%
F1 Score: 85.10%



Contributors

[Advaith B]
[Shreya Soni]
[Shaun Navanit Dcosta]
