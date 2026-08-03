from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# ===========================
# Load Model Files
# ===========================

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/label_encoders.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")


# ===========================
# Home Page
# ===========================

@app.route("/")
def home():

    df = pd.read_csv("dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

    total_employees = len(df)

    attrition_rate = round(
        (df["Attrition"] == "Yes").mean() * 100,
        2
    )

    avg_income = int(df["MonthlyIncome"].mean())

    total_departments = df["Department"].nunique()

    avg_job_satisfaction = round(
        df["JobSatisfaction"].mean(),
        2
    )

    return render_template(
        "index.html",
        total_employees=total_employees,
        attrition_rate=attrition_rate,
        avg_income=avg_income,
        total_departments=total_departments,
        avg_job_satisfaction=avg_job_satisfaction
    )


# ===========================
# Prediction
# ===========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = {

            "Age": int(request.form["Age"]),

            "BusinessTravel": request.form["BusinessTravel"],

            "Department": request.form["Department"],

            "DistanceFromHome": int(request.form["DistanceFromHome"]),

            "EnvironmentSatisfaction": int(request.form["EnvironmentSatisfaction"]),

            "JobRole": request.form["JobRole"],

            "JobSatisfaction": int(request.form["JobSatisfaction"]),

            "MonthlyIncome": int(request.form["MonthlyIncome"]),

            "OverTime": request.form["OverTime"],

            "TotalWorkingYears": int(request.form["TotalWorkingYears"]),

            "WorkLifeBalance": int(request.form["WorkLifeBalance"]),

            "YearsAtCompany": int(request.form["YearsAtCompany"])

        }

        input_df = pd.DataFrame([data])

        # Encode categorical columns
        for column in encoders:

            if column in input_df.columns:

                input_df[column] = encoders[column].transform(
                    input_df[column]
                )

        # Scale
        scaled_input = scaler.transform(input_df)

        # ===========================
        # Prediction
        # ===========================

        prediction = model.predict(scaled_input)[0]

        # Probability
        probability = model.predict_proba(scaled_input)

        confidence = round(
            probability.max() * 100,
            2
        )

        # ===========================================
        # Prediction Result + Risk Level
        # ===========================================

        if prediction == 1:

            prediction_text = "Employee is Likely to Leave"

            risk = "High"

            color = "red"

            recommendation = (
                "This employee has a high risk of leaving the company. "
                "HR should review workload, career growth, work-life balance, "
                "and compensation to improve employee retention."
            )

        else:

            prediction_text = "Employee is Likely to Stay"

            risk = "Low"

            color = "green"

            recommendation = (
                "This employee is likely to stay with the company. "
                "Continue employee engagement, recognize good performance, "
                "and provide regular career development opportunities."
            )

        return render_template(

            "result.html",

            prediction=prediction_text,

            probability=confidence,

            risk=risk,

            color=color,

            recommendation=recommendation

        )

    except Exception as e:

        return f"<h2>Error:</h2><br>{str(e)}"


# ===========================
# Run Flask
# ===========================

if __name__ == "__main__":

    app.run(debug=True)