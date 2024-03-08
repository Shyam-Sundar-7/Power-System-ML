# Power-System-ML

## Project Overview

The primary objective of this project is to develop a machine learning model capable of predicting the "Load_Type" of a power system based on historical data. The "Load_Type" categorization includes "Light_Load", "Medium_Load", and "Maximum_Load". This classification problem requires candidates to apply their skills in data preprocessing, exploratory data analysis (EDA), feature engineering, model selection, and model evaluation to predict the load type accurately.

## Project Progress

- **Exploratory Data Analysis (EDA):** The initial phase involved performing EDA on the dataset, which was documented in the _**EDA.ipynb**_ file. This involved formatting the Date_Time column appropriately, extracting NSM (Number of Seconds from Midnight) values to understand the timing, and creating plots to visualize the data distribution and relationships. Initial models were built based on three attributes, resulting in an accuracy of 77% shown in _**ml_model_bulding.ipynb**_ in validation dataset.

![result1](images/ml1.PNG)

- **Model Building and Evaluation:** Despite achieving decent performance on the training data, the models performed poorly on the validation dataset. To address this issue, additional attributes in the dataset were subjected to imputation and feature engineering. These enhancements were discussed in detail, along with plots and assumptions. Subsequently, a new model was built in the _**ml_model_building2.ipynb**_ notebook, resulting in an improved accuracy of 83% in validation dataset.

  - The test dataset yields a 69% accuracy with adaboost classifier by training the whole training and validation dataset.


![result1](images/ml2.PNG)


## Dataset Splitting

The test dataset consists of the last month of data from 2018. Out of the remaining months, 20% is allocated as the validation dataset, while the rest is used for training the model.

## Streamlit App run:


## Running the Streamlit Application

To run the Streamlit application:

1. Clone the repository:
   ```
   git clone https://github.com/Shyam-Sundar-7/Power-System-ML.git
   ```

2. Create a virtual environment and install the dependencies from requirements.txt.

    ```
    python -m venv env
    source env/bin/activate  # Activate the virtual environment
    pip install -r requirements.txt
    ```

3. run the following command in your terminal:

    ```
    streamlit run app.py
    ```


# Local Setup Output in Streamlit

[streamlit-app-2024-03-05-18-03-88.webm](https://github.com/Shyam-Sundar-7/Power-System-ML/assets/101181076/81451b68-f697-41bd-9c02-970e306ec0c4)
