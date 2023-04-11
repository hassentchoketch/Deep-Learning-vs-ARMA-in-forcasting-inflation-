import os 
import pandas as pd 
import matplotlib.pyplot as plt 
from helper import *


from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

from sklearn.metrics import mean_absolute_error, mean_squared_error


import warnings
warnings.filterwarnings('ignore')


cwd = os.getcwd()
inf_ = load_transform_data(data_file = 'DZ_Consumption_price_index.csv') # dataframe (300,1) 1-1-88 to 1-12-2022

inf = inf_.loc['1998-01-01':'2020-12-01','CPI(%)']
inf_test = inf_.loc['2021-01-01':'2022-12-01','CPI(%)']

# Appluing the adfuller test
results = adfuller(inf) 

inf_stationary = inf.diff().dropna()
results = adfuller(inf_stationary)  


# Searching over AIC and BIC
order_aic_bic = []
for p in range (1,13):
    for q in range(1,13):
        try:
            model = sm.tsa.arima.ARIMA(inf,order = (p,1,q))
            results = model.fit()
            order_aic_bic.append((p,q,results.aic,results.bic))
        except:  
            order_aic_bic.append((p,q,None,None))  
order_df = pd.DataFrame(order_aic_bic,columns=['p','q','aic','bic'])
order_df.to_csv(cwd + '\\results\\arima order sellction.csv')

# sort by aic and bic 
print("AIC:", order_df.sort_values('aic').iloc[0,:2])   
print("BIC:", order_df.sort_values('bic').iloc[0,:2]) 

arima_models = ['ARIMA-AIC','ARIMA-BIC']
metrics = ['mse','rmse','mae']
horizons= [6,12,18,24]
results = pd.DataFrame(columns=['model', 'horizon', 'metric', 'value'])

for model in arima_models:
    user_input_hyp_param = input(f"Please enter respectively the following optimal parameters for {model} model : p , q  separated by commas: ")
    # split the input into separate values
    values = user_input_hyp_param.split(",")
    # convert the values to the appropriate data type (e.g. int, float, str)
    param = []
    for value in values:
        value = value.strip()
        if value.isdigit():   
          param.append(int(value)) 
        elif "." in value and value.replace(".", "").isdigit(): 
          param.append(float(value)) 
        else:
          param.append(value) 


    model_ = sm.tsa.arima.ARIMA(inf,order = (param[0],1,param[1]),freq= 'MS')
    reg_results = model_.fit()
    resids = reg_results.resid
    
    

    fig,ax = plt.subplots(figsize =(8,8))
    ax = reg_results.plot_diagnostics()
    plt.savefig(cwd +f'\\results\\graphs\\{model}_models_diagnostics.png',dpi=300, bbox_inches='tight')
    # plt.show()



    # Forecasting 
    forecast = reg_results.get_forecast(steps =24)
    mean_forecast = forecast.predicted_mean  # from 

    confidence_intervals= forecast.conf_int()
    # print(confidence_intervals)
    
    # Generate forecasts for test data
    forecast = reg_results.forecast(steps=24)

# # Calculate MAE, MSE, RMSE, and MAPE of the forecasts
#     mae = mean_absolute_error(inf_test, forecast)
#     mse = mean_squared_error(inf_test, forecast)
#     rmse = np.sqrt(mse) 
#     print('MAE_model: ', mae)
#     print('MSE_model: ', mse)
#     print('RMSE_mode: ', rmse)
    
    for metric in metrics:
       for h in horizons:
         if metric == 'mse':
           loss = mean_squared_error(inf_test[:h], forecast[:h])
         elif metric == 'rmse':
           loss = np.sqrt(mean_squared_error(inf_test[:h], forecast[:h]))
         else :
           loss = mean_absolute_error(inf_test[:h], forecast[:h])
      
         results = results.append({'model': model, 'horizon': h, 'metric': metric, 'value': loss}, ignore_index=True)
    
  
    # metrics=['mse','rmse','ame']
    # for i in metrics:
    #      print(f'{i}_{model} :',loss_comp(mean_forecast, inf_test.values.reshape(-1),loss=i))

    fig2 = plt.figure(figsize =(8,8))
    # plot the original time series
    plt.plot(inf_test, label='Original Time Series')

    # plot the mean forecast
    plt.plot(mean_forecast, label='Mean Forecast')

    # plot the upper and lower confidence intervals
    plt.fill_between(x=mean_forecast.index, y1=confidence_intervals['lower CPI(%)'], 
                    y2=confidence_intervals['upper CPI(%)'], alpha=0.2, 
                    color='gray', label='Confidence Interval')
    # set plot title and axis labels
    plt.title(f'{model} Forecast of CPI(%)')
    plt.xlabel('Date')
    plt.ylabel('CPI(%)')

    # add legend to the plot
    plt.legend()
    # savefig
    fig2.savefig(cwd +f'\\results\\graphs\\Actual vs predicted values of {model} model (testing dataset).png',dpi=300, bbox_inches='tight')
    # display the plot
    # plt.show()
    
results.to_csv(cwd +f'\\results\\ARIMA forecasting performence.csv') 