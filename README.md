# KWS_Model_Training
This repository  is a hands-on deep learning workspace for (KWS) models on the Google Speech Commands dataset. Participants will incrementally upgrade their models from a an MLP to Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and efficient Depthwise-CNNs, benchmarking accuracy and parameter size along the way.

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

 ## The Contract
By participating in this session, you agree to:
1. Complete the architecture evolution tasks sequentially.
2. Log your model parameters and accuracy at every step.
3. Understand *why* an architecture fails before moving to the next upgrade.
4. Produce a final benchmark table comparing all your models.

##  Workspace Setup & Installation

To get started, you need to clone this repository into your local workspace and install the required dependencies.

### 1. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone https://github.com/RanimKhalfallah/KWS_Model_Training.git
cd KWS_Model_Training
```
### 1. Create a virtual environment
```bash
python -m venv kws_env
# On /Linux:
source kws_env/bin/activate
```
### 3. Install Dependencies
```bash
pip install tensorflow tensorflow-datasets numpy scikit-learn librosa matplotlib jupyterlab
```

### 4. Load your data
You can use the code injected in the second cell to load the "Google Speech Commands" dataset
Note :  it will download the ~2GB dataset.

### 5. Final Deliverable

At the end of the session, you must present a benchmark table comparing your 4 models:
Architecture,	Total Parameters,	Training Time (per epoch),	Validation Accuracy
Baseline MLP			
2D CNN			
LSTM			
DS-CNN	
