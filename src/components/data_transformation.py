import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer
@dataclass
class DataTransformationConfig:
    def __init__(self):
       self.preprocessor_obj_name :str= os.path.join('artifact','preprocessor.pkl')


class featuregenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bill_amt1_ind=8
        self.bill_amt2_ind=9
        self.bill_amt3_ind=10
        self.bill_amt4_ind=11
        self.bill_amt5_ind=12
        self.bill_amt6_ind=13
        self.pay_amt1_ind=14
        self.pay_amt2_ind=15
        self.pay_amt3_ind=16
        self.pay_amt4_ind=17
        self.pay_amt5_ind=18
        self.pay_amt6_ind=19

    def fit(self, x,y=None):
        return self
    
    def transform(self,x, y=None):
        slack_sept= x[:,self.bill_amt1_ind]- x[:, self.pay_amt1_ind]
        slack_aug=  x[:,self.bill_amt2_ind]- x[:, self.pay_amt2_ind]
        slack_july= x[:,self.bill_amt3_ind]- x[:, self.pay_amt3_ind]
        slack_jun=  x[:,self.bill_amt4_ind]- x[:, self.pay_amt4_ind]
        slack_may=  x[:,self.bill_amt5_ind]- x[:, self.pay_amt5_ind]
        slack_apr=  x[:,self.bill_amt6_ind]- x[:, self.pay_amt6_ind]

        x_new= np.c_[x[:,0:8],slack_sept, slack_aug, slack_july,slack_jun, slack_may,slack_apr]

        return x_new

class DataTransformation:
    def __init__(self):
        self.transform_config= DataTransformationConfig()

    def get_transformer_object(self):
        try:
            cat_features=['SEX', 'EDUCATION','MARRIAGE']
            num_features=['LIMIT_BAL','AGE','PAY_0','PAY_2','PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
                           'BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                            'PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            logging.info("Preparing num pipeline ")
            num_pipeline=Pipeline(steps=[("Imputer",SimpleImputer(strategy='median')),
                             ('featuregeneration', featuregenerator()),
                             ("Scaling", StandardScaler())])
            
            logging.info("Preparing cat pipeline ")
            cat_pipeline= Pipeline(steps=[("Imputer",SimpleImputer(strategy='most_frequent')), 
                               ("OnehotEncoding", OneHotEncoder(drop='first'))])
            
            logging.info("Preparing preprocessor object")
            preprocessor=ColumnTransformer([('num_pipeline',num_pipeline, num_features),
                                      ("Cat_pipeline",cat_pipeline,cat_features)])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try :
            logging.info(f"{'>>'*10}Starting Data Ingestion{'<<'*10}")
            logging.info("preparing train and test dataframes")
            train_df=pd.read_csv(train_data_path)
            test_df= pd.read_csv(test_data_path)

            x_train=train_df.drop(columns=['default payment next month'], axis=1)
            x_test=test_df.drop(columns=['default payment next month'], axis=1)

            y_train=train_df['default payment next month']
            y_test=test_df['default payment next month']

            logging.info("Calling get_transformer_object method")
            preprocessor_obj=self.get_transformer_object()

            preprocessor_obj.fit(x_train)
            x_train_preprocessed= preprocessor_obj.transform(x_train)
            x_test_preprocessed= preprocessor_obj.transform(x_test)
            logging.info(f"Shape of x_train_preprocessed is [{x_train_preprocessed.shape}]")
            logging.info(f"Shape of x_test_preprocessed is [{x_test_preprocessed.shape}]")

            train_arr=np.c_[x_train_preprocessed,np.asarray(y_train)]
            test_arr=np.c_[x_test_preprocessed,np.asarray(y_test)]
            
            logging.info("Saving the preprocessor object")
            save_object(self.transform_config.preprocessor_obj_name, preprocessor_obj)
            logging.info(f"{'=='*10}Data Transformation Completed{'=='*10}")
            return (
                train_arr,
                test_arr,
                self.transform_config.preprocessor_obj_name
            )

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=="__main__":
    dataingestion= DataIngestion()
    train_path, test_path= dataingestion.initiate_data_ingestion()

    datatransformation= DataTransformation()
    train_arr, test_arr, _ = datatransformation.initiate_data_transformation(train_path, test_path)

    model_trainer=ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)