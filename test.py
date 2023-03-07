import os
import pandas as pd
from helper import *

def main():
    cwd = os.getcwd()
    
    test = pd.read_csv(cwd+'\\results\\data\\testing_dataset.csv',index_col='date')
    model = models_loader()
    forecasts = model_forecast(model = model , series = test.values)
    
    date = pd.to_datetime(test.index)[12 - 1 :]
    actual = test[12 - 1 :]
    loss = loss_comp(forecasts, actual.values.reshape(-1))
    plot_series(date,(actual, forecasts),path= 'Actual vs predicted values (test)',legend=["Actual_values", "Predicted_values"],title=" Actual vs Forecasted values ",loss =loss)
    print('Mean Squared Error of prediction is:',loss)

if __name__ == "__main__":
    main()
