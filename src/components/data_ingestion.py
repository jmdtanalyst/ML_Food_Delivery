import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd


from dataclasses import (
    dataclass,
)  # dataclasses is a library used to create classes variables.
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")


# Class DataIngestion will load the variable 'DataIngestionConfig' paths.
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # Function initiate_data_ingestion will run scripts to collect the data.
    def initiate_data_ingestion(self):
        logging.info("Entering in  the data ingestion method")
        try:
            # Command to get the data (from csv, or json, or database)
            df = pd.read_csv("data\deliverytime.csv")

            # Save the data collected indo a path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Data ingestion completed succesfuly")
            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    raw_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        raw_data_path
    )

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
