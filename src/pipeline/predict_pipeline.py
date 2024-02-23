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
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            logging.ERROR(e, sys)
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        Delivery_person_Age: int,
        Delivery_person_Ratings: float,
        Type_of_order: str,
        Type_of_vehicle: str,
        distance: float,
    ):
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Type_of_order = Type_of_order
        self.Type_of_vehicle = Type_of_vehicle
        self.distance = distance

    logging.info("def get_data")

    def get_data_as_data_frame(self):
        try:
            logging.info("get_data_as_data_frame")
            custom_data_input_dict = {
                "Delivery_person_Age": [self.Delivery_person_Age],
                "Delivery_person_Ratings": [self.Delivery_person_Ratings],
                "Type_of_order": [self.Type_of_order],
                "Type_of_vehicle": [self.Type_of_vehicle],
                "distance": [self.distance],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
