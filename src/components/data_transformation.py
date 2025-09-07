import sys
from dataclasses import  dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.utils import save_object
from src.logger import logging
import os
import pickle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path =os.path.join('artifacts', 'processor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """Function responsible for data transformation"""

        try:
            numerical_columns = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
            categorical_columns = ['Soil Type', 'Crop Type']

            num_pipeline = Pipeline(
                steps =[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Categorical Columns: {categorical_columns}')
            logging.info(f'Numerical Columns: {numerical_columns}')

            preprocessor = ColumnTransformer([
                ('num_pipline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading Train and Test data Completed')

            logging.info('Obtaining processing object')

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Fertilizer Name"
            numerical_columns = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

            input_feature_train_df = train_df.drop(columns = ["Fertilizer Name"], axis =1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = train_df.drop(columns = ["Fertilizer Name"], axis =1)
            target_feature_test_df = train_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)



            





