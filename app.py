from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


## Routers
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("predict.html")
    else:
        data = CustomData(
            Delivery_person_Ratings=request.form.get("Delivery_person_Ratings"),
            multiple_deliveries=request.form.get("multiple_deliveries"),
            distance=request.form.get("distance"),
            Weatherconditions=request.form.get("Weatherconditions"),
            Road_traffic_density=request.form.get("Road_traffic_density"),
            Type_of_vehicle=request.form.get("Type_of_vehicle"),
            Festival=request.form.get("Festival"),
            City=request.form.get("City"),
        )
        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(pred_df)
        print(results)
        print(results[0])
        return render_template("predict.html", results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
