
from flask import Flask, request, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the saved clustering model and preprocessing objects
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    with open(MODEL_PATH, "rb") as file:
        model_data = pickle.load(file)

    transformer = model_data["transformer"]
    scaler = model_data["scaler"]
    pca = model_data["pca"]
    kmeans = model_data["kmeans"]

    print("SmartCart model loaded successfully!")

except Exception as e:
    print("Error loading model.pkl:", e)
    raise


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read customer information from the HTML form
        customer_data = {
            "Income": float(request.form["Income"]),
            "Age": float(request.form["Age"]),
            "Recency": float(request.form["Recency"]),
            "Total_Spending": float(request.form["Total_Spending"]),
            "Total_Children": int(request.form["Total_Children"]),
            "Education": request.form["Education"],
            "Living_with": request.form["Living_with"],
            "Customer_Tenure_Days": float(
                request.form["Customer_Tenure_Days"]
            ),
            "NumWebVisitsMonth": int(
                request.form["NumWebVisitsMonth"]
            ),
            "NumWebPurchases": int(
                request.form["NumWebPurchases"]
            ),
            "Complain": int(request.form["Complain"]),
            "NumCatalogPurchases": int(
                request.form["NumCatalogPurchases"]
            ),
            "NumDealsPurchases": int(
                request.form["NumDealsPurchases"]
            ),
            "Response": int(request.form["Response"]),
            "NumStorePurchases": int(
                request.form["NumStorePurchases"]
            )
        }

        # Convert input into a DataFrame
        customer_df = pd.DataFrame([customer_data])

        # Check the columns expected by the fitted transformer
        expected_columns = list(transformer.feature_names_in_)
        missing_columns = [
            column for column in expected_columns
            if column not in customer_df.columns
        ]

        if missing_columns:
            raise ValueError(
                "The saved transformer requires additional columns: "
                + ", ".join(missing_columns)
                + ". The input fields and model.pkl must use the same "
                  "features as the training data."
            )

        # Match the original training column order
        customer_df = customer_df[expected_columns]

        # Apply the same preprocessing used during training
        encoded_data = transformer.transform(customer_df)
        scaled_data = scaler.transform(encoded_data)
        pca_data = pca.transform(scaled_data)

        # Predict customer segment
        cluster = int(kmeans.predict(pca_data)[0])

        prediction_text = (
            f"This customer belongs to Cluster {cluster}."
        )

        return render_template(
            "index.html",
            prediction_text=prediction_text
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Unable to predict: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)
