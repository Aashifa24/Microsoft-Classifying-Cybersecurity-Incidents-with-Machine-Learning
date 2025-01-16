# Microsoft-Classifying-Cybersecurity-Incidents-with-Machine-Learning


### **Project Overview**  
This project focuses on developing a machine learning model to predict the triage grade of cybersecurity incidents (True Positive, Benign Positive, or False Positive). The goal is to enhance the efficiency of **Security Operation Centers (SOC)** by automating the triage process and improving decision-making for incident management.

### **Problem Statement**  
**Security Operation Centers (SOCs)** face challenges with high volumes of cybersecurity incidents. The manual triage process is time-consuming and prone to human errors. This project leverages machine learning to automate triage, reducing effort and improving accuracy.

### **Approach**  
- Conducted **Exploratory Data Analysis (EDA)** to understand the dataset and preprocess the data.  
- Balanced the dataset using **SMOTE (Synthetic Minority Oversampling Technique)** to address class imbalance.  
- Selected **top-performing features** for model training using feature selection methods.  
- Trained and evaluated multiple machine learning models:  
  - **Logistic Regression**  
  - **Decision Tree**  
  - **Random Forest**  
  - **LightGBM**  
  - **XGBoost** (selected as the final model)  
- Fine-tuned hyperparameters using **RandomizedSearchCV** to improve performance.  
- Chose **XGBoost** as the final model based on the best performance metrics.

### **Key Results**  
- **Best Model**: **XGBoost**  
- **Accuracy**: **90%**  
- **Macro Average F1-score**: **90%**  
- **Weighted Average F1-score**: **91%**

### **Files in the Repository**  
- **Data_preprocessing_and_EDA.ipynb**: Notebook for data cleaning, preprocessing, and exploratory data analysis.  
- **Model_building_and_Evaluation.ipynb**: Notebook for model training, hyperparameter tuning, and evaluation.  
- **best_XGB_Classification_model_without_SMOTE.pkl**: Serialized file of the best **XGBoost** model trained without SMOTE.  
- **best_xgboost_Classification_model_SMOTE.pkl**: Serialized file of the best **XGBoost** model trained with SMOTE.  
- **documentation.pdf**: PDF containing detailed project documentation. [Click here to view the project documentation](https://link_to_documentation)  
- **presentation.pptx**: PowerPoint presentation summarizing the project. [Click here to view the project presentation](https://link_to_presentation)

### **Steps to Reproduce**

1. **Clone the repository**:  
   `git clone <repository_url>`

2. **Navigate to the project folder**:  
   `cd <project_folder>`

3. **Open the Jupyter notebooks**:  
   - **Data_preprocessing_and_EDA.ipynb** for data preparation and preprocessing.  
   - **Model_building_and_Evaluation.ipynb** for training and evaluation.

4. **Run the cells in sequence** to reproduce the results.  
   Ensure that the model is trained using the provided steps, and save the best model as .pkl files for future use.

5. **Load the saved model (.pkl files) to make predictions on new data**:  
```python  
import pickle  
# Load the model  
model = pickle.load(open('best_xgboost_Classification_model_SMOTE.pkl', 'rb'))
```

### **Requirements**  
Install the required Python libraries by running:  
`pip install -r requirements.txt`

### **Future Enhancements**  
- Expand the dataset for broader generalization.  
- Experiment with additional **feature engineering techniques**.  
- Automate the deployment of the model into **SOC workflows** for real-time incident triage.




