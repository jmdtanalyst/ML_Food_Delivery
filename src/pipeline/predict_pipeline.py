import sys
import os
import pandas as pd
from src.exception import CustomException
from src.exception import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            logging.ERROR(e, sys)
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        Delivery_person_Ratings: float,
        multiple_deliveries: int,
        distance: float,
        Weatherconditions: str,
        Road_traffic_density: str,
        Type_of_vehicle: str,
        Festival: str,
        City: str,
    ):

        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.multiple_deliveries = multiple_deliveries
        self.distance = distance
        self.Weatherconditions = Weatherconditions
        self.Road_traffic_density = Road_traffic_density
        self.Type_of_vehicle = Type_of_vehicle
        self.Festival = Festival
        self.City = City

    logging.info("def get_data")

    def get_data_as_data_frame(self):
        try:
            logging.info("get_data_as_data_frame")
            custom_data_input_dict = {
                "Delivery_person_Ratings": [self.Delivery_person_Ratings],
                "multiple_deliveries": [self.multiple_deliveries],
                "distance": [self.distance],
                "Weatherconditions": [self.Weatherconditions],
                "Road_traffic_density": [self.Road_traffic_density],
                "Type_of_vehicle": [self.Type_of_vehicle],
                "Festival": [self.Festival],
                "City": [self.City],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
