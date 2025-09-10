# FertiSense

FertiSense is a machine learning-powered web application designed to predict optimal fertilizer recommendations based on soil and crop parameters. Built with Flask, it leverages advanced models (CatBoost, XGBoost, and scikit-learn) to provide actionable insights for farmers and agronomists.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training & Artifacts](#model-training--artifacts)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
FertiSense aims to optimize fertilizer usage by analyzing key soil and crop metrics. Users input parameters such as temperature, humidity, moisture, soil type, crop type, and nutrient levels. The app processes these inputs and predicts the best fertilizer recommendation using a trained machine learning model.

## Features
- **Web Interface**: User-friendly Flask web app for input and results display.
- **ML Pipeline**: Automated data preprocessing and prediction using saved models and processors.
- **Custom Data Handling**: Accepts various soil and crop types, with robust input validation.
- **Extensible**: Easily adaptable for new crops, soil types, or additional features.

## Installation
### Prerequisites
- Python 3.7+
- pip

### Clone the Repository
```powershell
git clone https://github.com/ayookuns1/FertiSense.git
cd FertiSense
```

### Install Dependencies
```powershell
pip install -r requirements.txt
```

## Usage
### Run the Application
```powershell
python app.py
```
The app will start on `http://0.0.0.0:5000/` by default.

### Web Interface
1. Go to the home page (`/`).
2. Enter the required parameters:
	- Temperature
	- Humidity
	- Moisture
	- Soil Type
	- Crop Type
	- Nitrogen
	- Potassium
	- Phosphorous
3. Submit the form to receive fertilizer recommendations.

## Project Structure
```
FertiSense/
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── setup.py              # Package setup
├── artifacts/            # Model and processor files
│   ├── model.pkl
│   ├── processor.pkl
│   └── ...
├── src/
│   └── pipeline/
│       └── predict_pipeline.py
├── templates/            # HTML templates
├── logs/                 # Log files
├── notebooks/            # Jupyter notebooks for EDA
└── README.md             # Project documentation
```

## Model Training & Artifacts
- Models are trained and saved in the `artifacts/` directory.
- Preprocessing steps are serialized for consistent predictions.
- Training scripts and EDA are available in the `notebooks/` folder.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss your ideas.

## License
This project is licensed under the MIT License.

## Author
Ayoola Okunlola ([ayookuns830@gmail.com](mailto:ayookuns830@gmail.com))