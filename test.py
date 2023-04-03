import os
import pandas as pd
from helper import *

def main():
    cwd = os.getcwd()
    
    test = pd.read_csv(cwd+'\\results\\data\\testing_dataset.csv',index_col='date',parse_dates=['date'])
    
    models  = ['LSTM','DNN','RNN','CNN'] 
    
    for model in models:
        
        model_ = models_loader(model)
        forecasts = model_forecast(model = model_ , series = test.values)
        
        actual = test[12 - 1 :]
        loss = loss_comp(forecasts, actual.values.reshape(-1))
        plot_series((test.index)[12 - 1 :],(actual, forecasts),fig_name= f'Actual vs predicted values of {model} model (testing_dataset)',legend=["Actual_values", "Predicted_values"],title=f" Actual vs predicted values of {model} (testing dataset) ",loss =loss)
        print(f'Mean Squared Error of prediction of {model} is:',loss)

if __name__ == "__main__":
    main()
