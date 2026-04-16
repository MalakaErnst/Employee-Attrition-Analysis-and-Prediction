@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None
    risk = None

    shap_labels = []
    shap_values = []
    input_values = {}

    if request.method == "POST":

        # Collect form inputs
        for feature in FEATURES:
            input_values[feature] = request.form.get(feature)

        df = pd.DataFrame([input_values])

        # Convert numeric columns safely
        for col in NUMERIC_FEATURES:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        # -----------------------------
        # TRANSFORM DATA
        # -----------------------------
        X_transformed = preprocessor.transform(df)

        # -----------------------------
        # SHAP (SAFE VERSION)
        # -----------------------------
        try:
            explainer = shap.TreeExplainer(final_model)
            shap_vals = explainer.shap_values(X_transformed)

            # binary classification handling
            if isinstance(shap_vals, list):
                shap_contrib = shap_vals[1][0]
            else:
                shap_contrib = shap_vals[0]

            feature_names = preprocessor.get_feature_names_out()

            top_shap = sorted(
                zip(feature_names, shap_contrib),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]

            shap_labels = [str(x[0]) for x in top_shap]
            shap_values = [round(float(x[1]), 4) for x in top_shap]

            print("SHAP:", list(zip(shap_labels, shap_values)))

        except Exception as e:
            print("SHAP ERROR:", e)
            shap_labels = []
            shap_values = []

        # -----------------------------
        # PREDICTION
        # -----------------------------
        probability = model.predict_proba(df)[0][1]
        prediction = int(probability >= 0.35)

        if probability < 0.25:
            risk = "Low"
        elif probability < 0.55:
            risk = "Moderate"
        elif probability < 0.75:
            risk = "High"
        else:
            risk = "Critical"

    return render_template(
        "index.html",
        cat_features=CATEGORY_MAP,
        num_features=NUMERIC_FEATURES,
        prediction=prediction,
        probability=probability,
        input_values=input_values,
        risk=risk,
        shap_labels=shap_labels,   # ✅ added
        shap_values=shap_values    # ✅ added
    )
