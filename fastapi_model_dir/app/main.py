import pandas as pd
from pickle import load

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field
from typing import List, Union

imputer = load(open('/app/imputer.pkl', 'rb'))
std_scaler = load(open('/app/scaler.pkl', 'rb'))

statsmodel_final_logit = sm.load('statsmodel_final_logit.pickle')
logit_params = statsmodel_final_logit.params.index.tolist()


numerical_cols = ['x12', 'x44', 'x53', 'x56', 'x58', 'x62', 'x91']
dummy_cols = ['x5', 'x31', 'x81']

phat_threshold = 0.75

#feature engineering, including applying scaler/imputer from training dataset, string processing/casting, dummy col creation, etc
def _input_preprocess(input):
  #1. Fixing the money and percents#
    input['x12'] = input['x12'].astype(str)
    input['x12'] = input['x12'].str.replace('$','')
    input['x12'] = input['x12'].str.replace(',','')
    input['x12'] = input['x12'].str.replace(')','')
    input['x12'] = input['x12'].str.replace('(','-')

    #coerce string columns in numerical_cols list to float, with NA for errors
    #input['x12'] = input['x12'].astype(float)
    input[numerical_cols] = input[numerical_cols].apply(pd.to_numeric, errors='coerce')


    #scale and impute numerical columns then concat back to string/categorical/discrete columns
    #any NA values from failed string to float casting will be giving the mean imputation from training data set
    imputed_df = pd.DataFrame(imputer.transform(input[numerical_cols]), columns=numerical_cols)
    imputed_std_df = pd.DataFrame(std_scaler.transform(imputed_df), columns=numerical_cols)


    input = pd.concat([imputed_std_df, input[dummy_cols].reset_index(drop=True)], axis=1, sort=False)


    input['x31_asia'] = 0
    input.loc[lambda df: df['x31'].str.strip().str.lower() == 'asia', 'x31_asia'] = 1

    input['x31_japan'] = 0
    input.loc[lambda df: df['x31'].str.strip().str.lower() == 'japan', 'x31_japan'] = 1

    input['x31_germany'] = 0
    input.loc[lambda df: df['x31'].str.strip().str.lower() == 'germany', 'x31_germany'] = 1

    ##

    input['x5_monday'] = 0
    input.loc[lambda df: df['x5'].str.strip().str.lower() == 'monday', 'x5_monday'] = 1

    input['x5_tuesday'] = 0
    input.loc[lambda df: df['x5'].str.strip().str.lower() == 'tuesday', 'x5_tuesday'] = 1

    input['x5_saturday'] = 0
    input.loc[lambda df: df['x5'].str.strip().str.lower() == 'saturday', 'x5_saturday'] = 1

    input['x5_sunday'] = 0
    input.loc[lambda df: df['x5'].str.strip().str.lower() == 'sunday', 'x5_sunday'] = 1

    ##

    input['x81_January'] = 0
    input.loc[lambda df: df['x81'].str.strip().str.lower() == 'january', 'x81_January'] = 1

    input['x81_February'] = 0
    input.loc[lambda df: df['x81'].str.strip().str.lower() == 'february', 'x81_February'] = 1

    input['x81_March'] = 0
    input.loc[lambda df: df['x81'].str.strip().str.lower() == 'march', 'x81_March'] = 1

    input['x81_May'] = 0
    input.loc[lambda df: df['x81'].str.strip().str.lower() == 'may', 'x81_May'] = 1

    input['x81_June'] = 0
    input.loc[lambda df: df['x81'].str.strip().str.lower() == 'june', 'x81_June'] = 1

    input['x81_July'] = 0
    input.loc[lambda df: df['x81'].str.strip().str.lower() == 'july', 'x81_July'] = 1

    input['x81_August'] = 0
    input.loc[lambda df: df['x81'].str.strip().str.lower() == 'august', 'x81_August'] = 1

    input['x81_September'] = 0
    input.loc[lambda df: df['x81'].str.strip().str.lower() == 'september', 'x81_September'] = 1

    input['x81_October'] = 0
    input.loc[lambda df: df['x81'].str.strip().str.lower() == 'october', 'x81_October'] = 1

    input['x81_November'] = 0
    input.loc[lambda df: df['x81'].str.strip().str.lower() == 'november', 'x81_November'] = 1

    input['x81_December'] = 0
    input.loc[lambda df: df['x81'].str.strip().str.lower() == 'december', 'x81_December'] = 1


    #remove original categorical columns after dummy columns are created in dataframe
    input = input.drop(columns = dummy_cols)
    
    return input

#return model prediction for a given input from the POST() call
def get_model_response(input):

    #input can either be one record in a JSON-string or multiple records in an array JSON-string
    if isinstance(input, list):
      #https://stackoverflow.com/questions/61814887/how-to-convert-a-list-of-pydantic-basemodels-to-pandas-dataframe
      try: 
        X = pd.DataFrame([i.__dict__ for i in input])
      except:
        X = pd.DataFrame([pd.json_normalize(i.__dict__) for i in input])
    else:
      X = pd.json_normalize(input.__dict__)
      
    X = X[dummy_cols + numerical_cols]

    X = _input_preprocess(X)

    total_rows = len(X)
    batch_size = 100000

    #run predictions in batches if input dataframe is large enough to possibly cause memory issues
    if round(total_rows/batch_size) >= 2:
      iter_res = pd.DataFrame()

      for i in range(0, total_rows, batch_size):
        # Calculate the starting and ending indices for the current batch
        start_index = i
        end_index = min(i + batch_size, total_rows)

        # Select the current batch of rows using iloc
        batch_df = X.iloc[start_index:end_index].copy()

        batch_df['phat'] = statsmodel_final_logit.predict(batch_df[logit_params])
        iter_res = pd.concat([iter_res, batch_df])
      X = iter_res

    else:
      X['phat'] = statsmodel_final_logit.predict(X[logit_params])


    X['business_outcome'] = 0
    X.loc[lambda df: df['phat'] >= phat_threshold, 'business_outcome'] = 1

    X = X.sort_index(axis=1)

    result = X.to_dict(orient="records")

    return result


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# pydantic models

class ModelInput(BaseModel):
    x5: str #| None = None
    x12: str #| float
    x31: str #| None = None
    x44: str #| float
    x53: str #| float
    x56: str #| float
    x58: str #| float
    x62: str #| float
    x81: str #| None = None
    x91: str #| float #



class ModelOutput(BaseModel):
    business_outcome: int | float
    phat: float
    x12: int | float
    x31_asia: int | float
    x31_germany: int | float
    x31_japan: int | float
    x44: int | float
    x53: int | float
    x56: int | float
    x58: int | float
    x5_monday: int | float
    x5_saturday: int | float
    x5_sunday: int | float
    x5_tuesday: int | float
    x62: int | float
    x81_August: int | float
    x81_December: int | float
    x81_February: int | float
    x81_January: int | float
    x81_July: int | float
    x81_June: int | float
    x81_March: int | float
    x81_May: int | float
    x81_November: int | float
    x81_October: int | float
    x81_September: int | float
    x91: int | float


@app.get('/')
async def root():
    return {'hello': 'universe'}


@app.post("/predict", status_code=200, response_model = List[ModelOutput])
def get_prediction(input: Union[ModelInput, List[ModelInput]] ):
    response_object = get_model_response(input)

    return response_object
