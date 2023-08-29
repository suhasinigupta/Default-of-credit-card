from setuptools import setup, find_packages

def get_requirements(file_name):
    req_list= []
    with open(file_name) as file_obj:
        req_list= file_obj.readlines()
        return req_list.remove("-e .")

setup(
    name="Default of credit card",
    author="Suhasini Gupta",
    author_email="suhasinigupta31@gmail.com",
    version="0.0.1" ,
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages(),
    )