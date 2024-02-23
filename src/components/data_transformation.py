import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import haversine as hs
from src.utils import save_object
from sklearn.model_selection import train_test_split
import numpy as np


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Function get_data_transformer_object will run pipelines to perform  data transformation
    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        try:

            logging.info("Applying StandardScaler on Numerical features:")

            # num_pipeline will handle numerical features

            num_pipeline = Pipeline(
                steps=[
                    # this step will transform the data by applying standardscaler
                    ("scaler", StandardScaler()),
                ]
            )

            logging.info(f"Applying one_hot_encoder on Categorrical Features")
            # cat_pipeline will handle categorical features
            cat_pipeline = Pipeline(
                steps=[
                    # this step will encoder the data by applying LabelEncoder
                    ("one_hot_encoder", OneHotEncoder()),
                    # this step will transform the data by applying standardscaler
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )
            return preprocessor
        except Exception as e:
            logging.info(CustomException(e, sys))
            raise CustomException(e, sys)

    # Function to start data transformation
    def initiate_data_transformation(self, raw_data_path):

        try:

            logging.info("DataTransformation - Reading data")
            df = pd.read_csv(raw_data_path)

            # Data Cleaning
            df.drop_duplicates(inplace=True)
            df.replace({"NaN": np.nan}, regex=True, inplace=True)
            df.dropna(inplace=True)
            df["Time_taken(min)"] = df["Time_taken(min)"].str.split().str[1]

            df.rename(columns={"Time_taken(min)": "Time_taken_min"}, inplace=True)
            df["Time_taken_min"] = pd.to_numeric(df["Time_taken_min"]).astype(int)
            df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"]).astype(
                int
            )
            df["Delivery_person_Ratings"] = pd.to_numeric(
                df["Delivery_person_Ratings"]
            ).astype(float)

            ### Swap negative localization to positive
            df["Restaurant_latitude"] = abs(df["Restaurant_latitude"])
            df["Restaurant_longitude"] = abs(df["Restaurant_longitude"])

            df = df[df["Restaurant_latitude"] > 0]

            def distance(rest_lat, rest_log, dest_lat, dest_long):
                rest = (rest_lat, rest_log)
                dest = (dest_lat, dest_long)

                return hs.haversine(rest, dest)

            # using lambda to apply the function distance, also rounding the distance
            df["distance"] = df.apply(
                lambda row: round(
                    distance(
                        row["Restaurant_latitude"],
                        row["Restaurant_longitude"],
                        row["Delivery_location_latitude"],
                        row["Delivery_location_longitude"],
                    )
                ),
                axis=1,
            )

            df = df[
                [
                    "Delivery_person_Ratings",
                    "Time_Order_picked",
                    "Weatherconditions",
                    "Road_traffic_density",
                    "Type_of_vehicle",
                    "multiple_deliveries",
                    "Festival",
                    "City",
                    "distance",
                    "Time_taken_min",
                ]
            ]

            ## preprocessing object

            logging.info("Obtaining preprocessing object")

            target_column_name = "Time_taken_min"

            numerical_columns = [
                "Delivery_person_Ratings",
                "multiple_deliveries",
                "distance",
            ]
            categorical_columns = [
                "Weatherconditions",
                "Road_traffic_density",
                "Type_of_vehicle",
                "Festival",
                "City",
            ]

            try:
                for feature in categorical_columns:
                    df[feature] = df[feature].str.strip()
            except Exception as e:
                print(e)

            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)

            os.makedirs(
                os.path.dirname(self.data_transformation_config.train_data_path),
                exist_ok=True,
            )
            os.makedirs(
                os.path.dirname(self.data_transformation_config.test_data_path),
                exist_ok=True,
            )

            train_set.to_csv(
                self.data_transformation_config.train_data_path,
                index=False,
                header=True,
            )
            test_set.to_csv(
                self.data_transformation_config.test_data_path, index=False, header=True
            )

            train_set = pd.read_csv(self.data_transformation_config.train_data_path)

            test_set = pd.read_csv(self.data_transformation_config.test_data_path)

            preprocessing_obj = self.get_data_transformer_object(
                numerical_columns, categorical_columns
            )

            input_feature_train_df = train_set.drop(
                columns=[target_column_name], axis=1
            )

            target_feature_train_df = train_set[target_column_name]

            input_feature_test_df = test_set.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_set[target_column_name]

            logging.info(f"Applying preprocessing on training dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            logging.info(f"Applying preprocessing on testing dataframe.")
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # uses the function 'save_object' on utils.py to save the preprocessing object in the file path
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.info(CustomException(e, sys))
            raise CustomException(e, sys)
