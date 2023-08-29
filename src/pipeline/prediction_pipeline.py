import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifact','preprocessor.pkl')
            model_path=os.path.join('artifact','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 LIMIT_BAL:int,
                 SEX: int,
                 EDUCATION:int,
                 MARRIAGE:int,
                 AGE:int,
                 PAY_0:int,PAY_2:int, PAY_3:int, PAY_4:int, PAY_5:int, PAY_6:int,
                 BILL_AMT1:int, BILL_AMT2: int, BILL_AMT3: int, BILL_AMT4: int, BILL_AMT5: int, BILL_AMT6: int,
                 PAY_AMT1:int, PAY_AMT2:int,PAY_AMT3:int,PAY_AMT4:int,PAY_AMT5:int,PAY_AMT6:int,
                ):
        
        self.limit_bal=LIMIT_BAL
        self.education= EDUCATION
        self.sex= SEX
        self.marriage=MARRIAGE
        self.age=AGE
        self.pay_at_sept= PAY_0 
        self.pay_at_aug=PAY_2 
        self.pay_at_july=PAY_3
        self.pay_at_jun=PAY_4 
        self.pay_at_may=PAY_5 
        self.pay_at_apr= PAY_6
        self.bill_amt_sept=BILL_AMT1
        self.bill_amt_aug=BILL_AMT2
        self.bill_amt_july=BILL_AMT3
        self.bill_amt_jun=BILL_AMT4
        self.bill_amt_may=BILL_AMT5
        self.bill_amt_apr=BILL_AMT6
        self.paid_at_sept=PAY_AMT1 
        self.paid_at_aug=PAY_AMT2 
        self.paid_at_july=PAY_AMT3
        self.paid_at_jun=PAY_AMT4
        self.paid_at_may=PAY_AMT5
        self.paid_at_apr=PAY_AMT6
       

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'LIMIT_BAL':[self.limit_bal],
                'SEX':[self.sex],
                'EDUCATION':[self.education],
                'MARRIAGE':[self.marriage],
                'AGE':[self.age],
                'PAY_0':[self.paid_at_sept],
                'PAY_2':[self.paid_at_aug],
                'PAY_3':[self.paid_at_july],
                'PAY_4':[self.paid_at_jun],
                'PAY_5':[self.paid_at_may],
                'PAY_6':[self.paid_at_apr],
                'BILL_AMT1': [self.bill_amt_sept],
                'BILL_AMT2': [self.bill_amt_aug],
                'BILL_AMT3': [self.bill_amt_july],
                'BILL_AMT4': [self.bill_amt_jun],
                'BILL_AMT5': [self.bill_amt_may],
                'BILL_AMT6': [self.bill_amt_apr],
                'PAY_AMT1':[self.paid_at_sept],
                'PAY_AMT2':[self.paid_at_aug],
                'PAY_AMT3':[self.paid_at_july],
                'PAY_AMT4':[self.paid_at_jun],
                'PAY_AMT5':[self.paid_at_may],
                'PAY_AMT6':[self.paid_at_apr],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)