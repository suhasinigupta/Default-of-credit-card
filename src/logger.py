import logging
import os
from datetime import datetime

ROOT_DIR= os.getcwd()
FILE_NAME= f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
FILE_PATH= os.path.join(ROOT_DIR,"logs", FILE_NAME)
os.makedirs(os.path.dirname(FILE_PATH), exist_ok= True)
 
logging.basicConfig(filename=FILE_PATH,
                    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
                    level=logging.INFO  )
