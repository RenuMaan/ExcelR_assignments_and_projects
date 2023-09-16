#!/usr/bin/env python
# coding: utf-8

# In[46]:


# libreries

from matplotlib.figure import Figure
import pandas as pd
import datetime
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

Data = pd.read_csv("Website Vistiors Daywise - Sheet1.csv")
pd.set_option('display.max_rows', None)

#changing from dd-mm-yyyy to yyyy-mm-dd
import datetime
date = list()
for d in Data.Date.tolist():
    t = datetime.datetime.strptime(d, "%d-%m-%Y").strftime("%Y-%m-%d")
    date.append(t)
    
Data['date'] = date
Data.drop("Date", axis =1, inplace =True)
Data["date"]= pd.to_datetime(Data["date"])
Data.columns = ["visitors","date"]
df = Data.set_index("date")

#repacing the outliers with median

df['visitors'] = np.where(df['visitors'] > 3934.4, 2751, df['visitors'])


# In[72]:


# Auto regression model
#splitting the time series obtained after transformation
def final_model(forecast_steps):
    X = df.visitors
    train, test = np.split(X, [int(.70 *len(X))])
    rmse_AR = float('inf')
    for lag in list(range(1,30)):
        model = AutoReg(train, lags=lag)
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(start = test.index[0],end = test.index[-1], dynamic=False)
        rmse = np.sqrt(np.abs((test-yhat)**2)).mean()
        if rmse<rmse_AR:
            rmse_AR = rmse
            predictions = yhat 
            lag = lag

    history = [value for value in X]
    #forecast_steps = 30
    yhat = []
    for i in range(1,forecast_steps):
        model_final = AutoReg(X, lags=lag)
        res = model_final.fit()
        output = res.forecast()
        yhat.append(output[0])
        new_row = pd.Series(output)
        X = X.append(new_row)  
    forecasted = pd.DataFrame(yhat, index = pd.date_range(df.index[-1],
                                                            df.index[-1] + pd.Timedelta(forecast_steps-2, unit='D')))
    forecasted.columns = ["forecast"]
    plt.figure(figsize = (8,3))
    plt.plot(df['visitors']) 
    plt.plot(forecasted)
    plt.xticks(rotation = 30)
    plt.legend(['history', 'forecast'],loc = 4)
    plt.ylabel('visitors')
    return {
         'my_df':forecasted,
         'my_plot': plt
         }


# In[ ]:




