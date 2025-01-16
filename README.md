# Microsoft-Classifying-Cybersecurity-Incidents-with-Machine-Learning


This project focuses on developing a machine learning model to predict the triage grade of cybersecurity incidents (True Positive, Benign Positive, or False Positive). The goal is to enhance the efficiency of Security Operation Centers (SOC) by automating the triage process and improving decision-making for incident management.

## Project Overview

### Problem Statement
Security Operation Centers (SOCs) face challenges with high volumes of cybersecurity incidents. The manual triage process is time-consuming and prone to human errors. This project leverages machine learning to automate triage, reducing effort and improving accuracy.

### Approach
1. Conducted **Exploratory Data Analysis (EDA)** to understand the dataset and preprocess the data.
2. Balanced the dataset using **SMOTE (Synthetic Minority Oversampling Technique)**.
3. Selected top-performing features for model training.
4. Trained and evaluated multiple machine learning models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - LightGBM
   - XGBoost
5. Fine-tuned hyperparameters using **RandomizedSearchCV**.
6. Chose XGBoost as the final model based on performance metrics.

### Key Results
- **Best Model:** XGBoost
- **Accuracy:** 90%
- **Macro Average F1-score:** 90%
- **Weighted Average F1-score:** 91%

## Files in the Repository

- **`Data_preprocessing_and_EDA.ipynb`:** Notebook for data cleaning, preprocessing, and exploratory data analysis.
- **`Model_building_and_Evaluation.ipynb`:** Notebook for model training, hyperparameter tuning, and evaluation.
- **`best_XGB_Classification_model_without_SMOTE.pkl`:** Serialized file of the best XGBoost model trained without SMOTE.
- **`best_xgboost_Classification_model_SMOTE.pkl`:** Serialized file of the best XGBoost model trained with SMOTE.
- **`documentation.pdf`:** PDF containing detailed project documentation. Link : [documentation.pdf](https://github.com/Aashifa24/Microsoft-Classifying-Cybersecurity-Incidents-with-Machine-Learning/blob/0e6bfb137794dbd990b26ee7b3110ab360f16139/documentation.pdf)
- **`presentation.pptx`:** PowerPoint presentation summarizing the project. Link : [https://github.com/Aashifa24/Microsoft-Classifying-Cybersecurity-Incidents-with-Machine-Learning/blob/fbeb5c814ff787b480b6e9c738b1360ea0ada23e/presentation.pptx
](https://github.com/Aashifa24/Microsoft-Classifying-Cybersecurity-Incidents-with-Machine-Learning/blob/80308646ff38d4a4c49f877982740954f0c8a94f/presentation.pptx)
## How to Run the Project

1. Clone the repository:
   ```
   git clone <repository_url>
   ```
2. Navigate to the project folder:
   ```
   cd <project_folder>
   ```
3. Open the Jupyter notebooks:
   - `Data_preprocessing_and_EDA.ipynb` for data preparation.
   - `Model_building_and_Evaluation.ipynb` for training and evaluation.
4. Run the cells in sequence to reproduce the results.
5. Load the saved model (`.pkl` files) to make predictions on new data.

## Requirements

Install the required Python libraries by running:
```
pip install -r requirements.txt
```

## Future Enhancements

- Expand the dataset for broader generalization.
- Experiment with additional feature engineering techniques.
- Automate the deployment of the model into SOC workflows.


