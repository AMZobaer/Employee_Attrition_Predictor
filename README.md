# End-to-End Employee Attrition Prediction using XGBoost

This project builds a machine learning system to predict employee attrition using the IBM HR Analytics dataset. The goal is to identify employees who are likely to leave the organization so that companies can take proactive retention measures.

## Project Overview

The project follows a complete machine learning workflow including:

- Data preprocessing and feature engineering
- Handling class imbalance
- Model comparison using Stratified Cross Validation
- Hyperparameter tuning
- Model evaluation using multiple performance metrics
- Deployment of the trained model using Gradio
- Hosting the application on Hugging Face Spaces

## Dataset

Dataset used: IBM HR Analytics Employee Attrition Dataset

Source:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

## Models Tested

Several machine learning models were evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost

Based on cross-validation performance, **XGBoost** was selected as the final model.

## Model Performance

Final model evaluation on the test dataset:

- Accuracy: ~0.83
- Precision: ~0.47
- Recall: ~0.47
- F1 Score: ~0.47
- ROC-AUC: ~0.78

These results indicate the model achieves a reasonable balance between precision and recall while handling class imbalance.

## Web Application

The trained model was deployed using **Gradio** to create an interactive interface where users can input employee attributes and receive attrition predictions.

Live Demo:
[AMZobaer/Employee_Attrition_Predictor
](https://huggingface.co/spaces/AMZobaer/Employee_Attrition_Predictor)

## Project Structure

Employee_Attrition_Predictor
│
├── notebooks
│   └── attrition_model_training.ipynb
│
├── app.py
├── xgb_model.pkl
├── requirements.txt
└── README.md

## Technologies Used

- Python
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Gradio
- Hugging Face Spaces

## How to Run the Project

1. Clone the repository
   git clone https://github.com/AMZobaer/Employee_Attrition_Predictor.git
2. Install dependencies
   pip install -r requirements.txt
3. Run the application
   python app.py

## Author
A M Zobaer
