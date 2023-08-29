import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, auc, confusion_matrix,accuracy_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        model_names=[]
        f1_score_list=[]
        recall_list=[]
        precision_list=[]
        accuracy_list=[]
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)
            f1= f1_score(y_test, y_test_pred)
            recall= recall_score(y_test, y_test_pred)
            precision= precision_score(y_test,y_test_pred)
            f1_score_list.append(f1)
            recall_list.append(recall)
            precision_list.append(precision)
            model_names.append(model)
            test_model_score = accuracy_score(y_test,y_test_pred)
            accuracy_list.append(test_model_score)
            report[list(models.keys())[i]] =  test_model_score
        analysis_df =pd.DataFrame({"Models": model_names,"FI SCORE":f1_score_list, "RECALL ":recall_list,"PRECISION":precision_list, "ACCURACY":accuracy_list})
        logging.info(f"{analysis_df}")
        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)