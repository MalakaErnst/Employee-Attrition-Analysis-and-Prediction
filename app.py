from flask import Flask, render_template, request
import pandas as pd
import joblib
import shap
import numpy as np

app = Flask(__name__)

# Load model once
model = joblib.load("attrition_model_calibrated.joblib")

print("Base pipeline type:", type(model))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    shap_data = None  # ✅ prevents crash on GET

    if request.method == "POST":
        try:
            # ----------------------------
            # 1. Collect form data
            # ----------------------------
            input_data = {
                "Age": int(request.form["Age"]),
                "BusinessTravel": request.form["BusinessTravel"],
                "DailyRate": int(request.form["DailyRate"]),
                "Department": request.form["Department"],
                "DistanceFromHome": int(request.form["DistanceFromHome"]),
                "Education": int(request.form["Education"]),
                "EducationField": request.form["EducationField"],
                "EnvironmentSatisfaction": int(request.form["EnvironmentSatisfaction"]),
                "Gender": request.form["Gender"],
                "HourlyRate": int(request.form["HourlyRate"]),
                "JobInvolvement": int(request.form["JobInvolvement"]),
                "JobLevel": int(request.form["JobLevel"]),
                "JobRole": request.form["JobRole"],
                "JobSatisfaction": int(request.form["JobSatisfaction"]),
                "MaritalStatus": request.form["MaritalStatus"],
                "MonthlyIncome": int(request.form["MonthlyIncome"]),
                "MonthlyRate": int(request.form["MonthlyRate"]),
                "NumCompaniesWorked": int(request.form["NumCompaniesWorked"]),
                "OverTime": request.form["OverTime"],
                "PercentSalaryHike": int(request.form["PercentSalaryHike"]),
                "PerformanceRating": int(request.form["PerformanceRating"]),
                "RelationshipSatisfaction": int(request.form["RelationshipSatisfaction"]),
                "StockOptionLevel": int(request.form["StockOptionLevel"]),
                "TotalWorkingYears": int(request.form["TotalWorkingYears"]),
                "TrainingTimesLastYear": int(request.form["TrainingTimesLastYear"]),
                "WorkLifeBalance": int(request.form["WorkLifeBalance"]),
                "YearsAtCompany": int(request.form["YearsAtCompany"]),
                "YearsInCurrentRole": int(request.form["YearsInCurrentRole"]),
                "YearsSinceLastPromotion": int(request.form["YearsSinceLastPromotion"]),
                "YearsWithCurrManager": int(request.form["YearsWithCurrManager"]),
            }

            input_df = pd.DataFrame([input_data])

            # ----------------------------
            # 2. Prediction
            # ----------------------------
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            prediction = "Attrition" if pred == 1 else "No Attrition"
            probability = round(prob * 100, 2)

            # ----------------------------
            # 3. SHAP (safe block)
            # ----------------------------
            try:
                # Transform input
                X_transformed = model.named_steps['preprocess'].transform(input_df)

                # Get classifier
                clf = model.named_steps['classifier']

                # SHAP explainer
                explainer = shap.TreeExplainer(clf)

                shap_vals = explainer.shap_values(X_transformed)

                # Binary classification fix
                shap_values = shap_vals[1][0]

                # Feature names after encoding
                feature_names = model.named_steps['preprocess'].get_feature_names_out()

                # Build shap_data
                shap_data = [
                    {"feature": name, "value": float(val)}
                    for name, val in zip(feature_names, shap_values)
                ]

                # Sort by importance
                shap_data = sorted(shap_data, key=lambda x: abs(x["value"]), reverse=True)[:10]

                print("SHAP DATA:", shap_data)

            except Exception as e:
                print("SHAP ERROR:", e)
                shap_data = None  # don't crash app

        except Exception as e:
            print("PREDICTION ERROR:", e)

    # ----------------------------
    # 4. Render
    # ----------------------------
    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        shap_data=shap_data
    )


# ----------------------------
# 5. Run locally (Render ignores this)
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
