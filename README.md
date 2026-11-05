<div align="center">

# Forest Fire Prediction

### An interactive machine learning dashboard for forest-fire classification

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Status-Academic%20Project-blue" alt="Project status">
</p>

<p><strong>Explore environmental data, understand fire-related patterns, and predict whether given conditions are associated with a forest fire.</strong></p>

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Dataset and Input Features](#dataset-and-input-features)
- [Machine Learning Model](#machine-learning-model)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running the Application](#running-the-application)
- [Running Tests](#running-tests)
- [Using the Dashboard](#using-the-dashboard)
- [Limitations](#limitations)
- [Contributors](#contributors)

## Overview

Forest Fire Prediction is a machine learning project that analyses environmental and meteorological conditions to classify whether a forest-fire event is associated with a given set of observations.

The project includes an interactive **Streamlit web application** built around a trained **Random Forest classification pipeline**. Alongside making predictions, the dashboard provides exploratory data analysis and model evaluation visualisations to help users understand the dataset and the factors associated with fire occurrence.

The application uses the Algerian Forest Fires dataset, with a cleaned dataset containing 243 observations.

> **Note:** This project is intended for academic and educational use. Its predictions should not be treated as an operational wildfire warning system or a substitute for official alerts and expert assessment.

## Key Features

- **Interactive dashboard:** Explore the project and its data through a Streamlit interface.
- **Exploratory data analysis:** Examine distributions and relationships between environmental variables.
- **Class distribution analysis:** Visualise the distribution of fire and non-fire observations.
- **Feature analysis:** Explore how input variables relate to the target class.
- **Regional analysis:** Compare observations across the dataset's two regions.
- **Model evaluation:** Review classification metrics, a confusion matrix, ROC-AUC, and feature importance.
- **Fire classification:** Enter environmental conditions and receive a model-generated fire/non-fire prediction.
- **Prediction probability:** View the model's estimated probability alongside the predicted class.

## How It Works

1. **Data input:** Environmental and meteorological measurements are collected from the dataset or entered through the prediction interface.
2. **Preprocessing:** Input features are arranged in the format expected by the trained model.
3. **Model inference:** The saved Random Forest pipeline processes the input and generates a classification.
4. **Result display:** The dashboard displays the predicted class and the model's estimated probability.
5. **Data exploration:** Users can inspect visualisations and evaluation results to better understand model behaviour.

```text
Environmental / Meteorological Inputs
                 |
                 v
          Input Validation
                 |
                 v
       Trained ML Pipeline
        (Random Forest)
                 |
                 v
        Fire Classification
                 |
          +------+------+
          |             |
          v             v
       Not Fire         Fire
          |             |
          +------+------+
                 |
                 v
       Prediction + Probability
          in Streamlit Dashboard
```

## Dataset and Input Features

The project uses a cleaned version of the **Algerian Forest Fires dataset**, containing 243 observations.

The prediction interface uses the following 13 input features:

| Feature | Description |
|---|---|
| Day | Day of the observation |
| Month | Month of the observation |
| Year | Year of the observation |
| Temperature | Temperature measurement |
| RH | Relative humidity |
| Ws | Wind speed |
| Rain | Rainfall measurement |
| FFMC | Fine Fuel Moisture Code |
| DMC | Duff Moisture Code |
| DC | Drought Code |
| ISI | Initial Spread Index |
| BUI | Buildup Index |
| Region | Region identifier |

The target represents the fire classification:

- `0` — Not Fire
- `1` — Fire

The FWI-related indices (FFMC, DMC, DC, ISI and BUI) are part of the Canadian Fire Weather Index system and represent different aspects of fuel moisture and fire behaviour.

## Machine Learning Model

The application uses a **Random Forest classifier** packaged as a reusable machine learning pipeline.

- **Algorithm:** Random Forest Classification
- **Preprocessing:** StandardScaler included in the saved pipeline
- **Saved model:** `models/fire_classifier.joblib`
- **Output:** Binary classification — Fire or Not Fire
- **Additional output:** Model-estimated class probability

Keeping preprocessing and the estimator together in a single pipeline ensures that prediction inputs are transformed consistently with the training workflow.

The dashboard also presents model evaluation information, including:

- Accuracy
- Precision
- Recall
- ROC-AUC
- Confusion matrix
- Feature importance

The actual metric values should be read from the dashboard or the model evaluation outputs rather than assumed from the algorithm.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing, model training and evaluation |
| Random Forest | Classification algorithm |
| Joblib | Saving and loading the trained model |
| Streamlit | Interactive web application |
| Matplotlib / Seaborn | Data visualisation |

## Repository Structure

```text
forest-fire-prediction/
│
├── app/
│   ├── app.py                  # Streamlit application entry point
│   ├── requirements.txt        # Python dependencies
│   └── ...                     # Supporting application files
│
├── dataset/                    # Dataset files
│
├── models/
│   └── fire_classifier.joblib  # Trained classification pipeline
│
├── src/                        # Source code for project workflows
│
├── outputs/                    # Generated plots and evaluation outputs
│
└── README.md                   # Project documentation
```

*The exact files inside each directory may vary as the project evolves. The virtual environment (`venv/` or `.venv/`) is local to each developer and should not be committed to Git.*

## Getting Started

Follow these steps to run the project locally.

### Prerequisites

- Python **3.10**
- Git
- pip
- A terminal (PowerShell, Command Prompt, macOS Terminal, or Linux shell)

Check your Python installation:

```bash
python --version
```

On Windows, if multiple Python versions are installed, you can check for Python 3.10 with:

```powershell
py -3.10 --version
```

### Installation on Windows

**1. Clone the repository**

```powershell
git clone https://github.com/spoorthy1202/forest-fire-prediction.git
cd forest-fire-prediction
```

**2. Navigate to the application directory**

```powershell
cd app
```

**3. Create a virtual environment**

```powershell
py -3.10 -m venv venv
```

**4. Activate the virtual environment**

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution with an execution-policy error, use this temporary session-only workaround:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\venv\Scripts\Activate.ps1
```

Alternatively, from Command Prompt (CMD), activate it with:

```cmd
venv\Scripts\activate.bat
```

**5. Install dependencies**

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Installation on macOS and Linux

**1. Clone the repository and enter the application directory**

```bash
git clone https://github.com/spoorthy1202/forest-fire-prediction.git
cd forest-fire-prediction/app
```

**2. Create and activate a virtual environment**

```bash
python3.10 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Application

After installing the dependencies and activating the virtual environment, run:

```bash
streamlit run app.py
```

Streamlit will start a local development server. Open the URL shown in your terminal, usually:

```text
http://localhost:8501
```

### Accessing the app from another device on the same network

To make the application accessible to other devices connected to the same Wi-Fi or local network, run:

```bash
streamlit run app.py --server.address 0.0.0.0
```

Then open the following address on the other device:

```text
http://<your-computer-local-IP>:8501
```

Replace `<your-computer-local-IP>` with your computer's local IPv4 address. Both devices must be on the same network, and your firewall must allow the connection.

This exposes the app on your local network; it does **not** make it publicly accessible over the internet. For public access, deploy the application to a suitable hosting platform and follow its security and configuration guidance.

## Running Tests

From the repository root, activate the appropriate virtual environment and run:

```bash
python -m pytest -q
```

The project's current test suite is expected to report:

```text
7 passed
```

If the test command cannot find `pytest`, install the project's test dependencies or install it separately:

```bash
python -m pip install pytest
```

## Using the Dashboard

1. Start the Streamlit application.
2. Open the local URL in your browser.
3. Explore the overview and exploratory data analysis sections.
4. Review the classification visualisations and model evaluation metrics.
5. Open the prediction interface.
6. Enter the 13 environmental and meteorological input values.
7. Submit the inputs to view the predicted class and estimated probability.

## Limitations

- The model is trained on a relatively small dataset (243 observations), so its predictions may not generalise to all forests, climates, or geographic regions.
- Model predictions depend on the quality, range, and accuracy of the supplied input values.
- A predicted probability is a model output, not a guarantee that a fire will or will not occur.
- The application is a machine learning demonstration and is not validated for emergency response, evacuation, or real-world fire prevention decisions.

## Contributors

Developed as an academic project at **Manipal Institute of Technology, Manipal Academy of Higher Education**.

- **Vatsal Jindal**
- **Akash V Paul**
- **Aditi Dayal**
- **Spoorthy S Deshpande**

**Project Guide:** Dr. Shankar Biradar

---

<div align="center">

If you find this project useful, consider giving the repository a ⭐.

**Forest Fire Prediction — Data-driven exploration and classification of forest-fire observations.**

</div>
