import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass


    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/processor.pkl'
            model= load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds  = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                Temperature: int,
                 Humidity: int,
                 Moisture: int,
                 Soil_Type: str,
                 Crop_Type: str,
                 Nitrogen: int,
                 Potassium: int,
                 Phosphorous: int):
        
        self.Temperature = Temperature
        self.Humidity = Humidity
        self.Moisture = Moisture
        self.Soil_Type = Soil_Type
        self.Crop_Type = Crop_Type
        self.Nitrogen = Nitrogen
        self.Potassium = Potassium
        self.Phosphorous = Phosphorous
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Temparature": [self.Temperature],
                "Humidity": [self.Humidity],
                "Moisture": [self.Moisture],
                "Soil Type": [self.Soil_Type],
                "Crop Type": [self.Crop_Type],
                "Nitrogen": [self.Nitrogen],
                "Potassium": [self.Potassium],
                "Phosphorous": [self.Phosphorous]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

