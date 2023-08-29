import pandas as pd
import os,sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    data_file_path: str=os.path.join('notebook','data','default of credit card clients.xls')
    train_data_path:str = os.path.join('artifact','data_ingestion','train.csv')
    test_data_path:str = os.path.join('artifact','data_ingestion','test.csv')
  
class DataIngestion:
  
  def __init__(self):
     self.data_ingestion_config=DataIngestionConfig()

  def initiate_data_ingestion(self):
     try :
        logging.info(f"{'>>'*10}Starting Data Ingestion {'<<'*10}")
        df= pd.read_excel(self.data_ingestion_config.data_file_path,skiprows=[0])
        logging.info("Splitting the data as train test")
        train_df, test_df = train_test_split(df, test_size=0.25)
        os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok= True)
        train_df.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
        test_df.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
        logging.info("Save the data in file path")
        logging.info(f"{'=='*10}Data Ingestion Completed{'=='*10}")
        return (self.data_ingestion_config.train_data_path, self.data_ingestion_config.test_data_path)
     except Exception as e:
         raise CustomException(e,sys)   

