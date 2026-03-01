import gradio as gr
import numpy as np
import pandas as pd
import pickle

# Load the pickled model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the prediction function
def predict_attrition(Age, BusinessTravel, DailyRate, Department, DistanceFromHome,
                    Education, EducationField, EnvironmentSatisfaction,
                    Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction,
                    MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked,
                    OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction,
                    StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear,
                    WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, 
                    YearsSinceLastPromotion, YearsWithCurrManager):
    
    # Convert categorical inputs to the format expected by the model
    Education_mapping = {"Below College": 1, "College": 2, "Bachelor": 3, "Master": 4, "Doctor": 5}
    EnvironmentSatisfaction_mapping = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
    JobInvolvement_mapping = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
    JobSatisfaction_mapping = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
    PerformanceRating_mapping = {"Low": 1, "Good": 2, "Excellent": 3, "Outstanding": 4}
    RelationshipSatisfaction_mapping = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
    WorkLifeBalance_mapping = {"Bad": 1, "Good": 2, "Better": 3, "Best": 4}
    
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'Age': [Age],
        'BusinessTravel': [BusinessTravel],
        'DailyRate': [DailyRate],
        'Department': [Department],
        'DistanceFromHome': [DistanceFromHome],
        'Education': [Education_mapping[Education]],
        'EducationField': [EducationField],
        'EnvironmentSatisfaction': [EnvironmentSatisfaction_mapping[EnvironmentSatisfaction]],
        'Gender': [Gender],
        'HourlyRate': [HourlyRate],
        'JobInvolvement': [JobInvolvement_mapping[JobInvolvement]],
        'JobLevel': [JobLevel],
        'JobRole': [JobRole],
        'JobSatisfaction': [JobSatisfaction_mapping[JobSatisfaction]],
        'MaritalStatus': [MaritalStatus],
        'MonthlyIncome': [MonthlyIncome],
        'MonthlyRate': [MonthlyRate],
        'NumCompaniesWorked': [NumCompaniesWorked],
        'OverTime': [OverTime],
        'PercentSalaryHike': [PercentSalaryHike],
        'PerformanceRating': [PerformanceRating_mapping[PerformanceRating]],
        'RelationshipSatisfaction': [RelationshipSatisfaction_mapping[RelationshipSatisfaction]],
        'StockOptionLevel': [StockOptionLevel],
        'TotalWorkingYears': [TotalWorkingYears],
        'TrainingTimesLastYear': [TrainingTimesLastYear],
        'WorkLifeBalance': [WorkLifeBalance_mapping[WorkLifeBalance]],
        'YearsAtCompany': [YearsAtCompany],
        'YearsInCurrentRole': [YearsInCurrentRole],
        'YearsSinceLastPromotion': [YearsSinceLastPromotion],
        'YearsWithCurrManager': [YearsWithCurrManager]
    })

    # Make the prediction
    prediction = model.predict(input_data)[0]
    return "Yes" if prediction == 1 else "No"

# Define the Gradio interface
app = gr.Interface(
    fn=predict_attrition,
    inputs=[
        gr.Number(label="Age", value=35),
            gr.Dropdown(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], label="Business Travel"),
            gr.Number(label="Daily Rate", value=800),
            gr.Dropdown(["Sales", "Research & Development", "Human Resources"], label="Department"),
            gr.Number(label="Distance From Home", value=10),
            gr.Dropdown(
                choices={"Below College": 1, "College": 2, "Bachelor": 3, "Master": 4, "Doctor": 5},
                label="Education"
            ),
            gr.Dropdown(
                ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
                label="Education Field"
            ),
            gr.Dropdown(
                choices={"Low": 1, "Medium": 2, "High": 3, "Very High": 4},
                label="Environment Satisfaction"
            ),
            gr.Radio(["Male", "Female"], label="Gender"),
            gr.Number(label="Hourly Rate", value=65),

            gr.Dropdown(
                choices={"Low": 1, "Medium": 2, "High": 3, "Very High": 4},
                label="Job Involvement"
            ),
            gr.Number(label="Job Level", value=2),
            gr.Dropdown(
                [
                    "Sales Executive", "Research Scientist", "Laboratory Technician",
                    "Manufacturing Director", "Healthcare Representative", "Manager",
                    "Sales Representative", "Research Director", "Human Resources"
                ],
                label="Job Role"
            ),
            gr.Dropdown(
                choices={"Low": 1, "Medium": 2, "High": 3, "Very High": 4},
                label="Job Satisfaction"
            ),
            gr.Radio(["Single", "Married", "Divorced"], label="Marital Status"),
            gr.Number(label="Monthly Income", value=6000),
            gr.Number(label="Monthly Rate", value=20000),
            gr.Slider(0, 9, step=1, label="Number of Companies Worked"),
            gr.Radio(["Yes", "No"], label="Over Time"),
            gr.Slider(11, 25, step=1, label="Percent Salary Hike"),
            gr.Dropdown(
                choices={"Low": 1, "Good": 2, "Excellent": 3, "Outstanding": 4},
                label="Performance Rating"
            ),
            gr.Dropdown(
                choices={"Low": 1, "Medium": 2, "High": 3, "Very High": 4},
                label="Relationship Satisfaction"
            ),
            gr.Slider(0, 3, step=1, label="Stock Option Level"),
            gr.Number(label="Total Working Years", value=10),
            gr.Slider(0, 6, step=1, label="Training Times Last Year"),
            gr.Dropdown(
                choices={"Bad": 1, "Good": 2, "Better": 3, "Best": 4},
                label="Work Life Balance"
            ),
            gr.Number(label="Years At Company", value=5),
            gr.Number(label="Years In Current Role", value=3),
            gr.Number(label="Years Since Last Promotion", value=1),
            gr.Number(label="Years With Current Manager", value=3),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Employee Attrition Prediction",
    description="Enter employee details to predict if they are likely to leave the company."
)

# Launch the interface
app.launch(share=True)