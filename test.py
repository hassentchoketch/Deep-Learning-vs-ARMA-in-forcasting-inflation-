import os
import pandas as pd
from helper import *

import warnings
warnings.filterwarnings('ignore')
tf.config.run_functions_eagerly(True)

def main():
    config_plot()
    cwd = os.getcwd()
    
    test = pd.read_csv(cwd+'\\results\\data\\testing_dataset.csv',index_col='date',parse_dates=['date'])
    
    models  = ['NN','MLP','simpl_RNN','LSTM','BI_LSTM','CNN' ] 
    metrics = ['rmse','mae']
    horizons= [6,12,18,24]
    
    results = pd.DataFrame(columns=['model', 'horizon', 'metric', 'value'])
   
    for model in models:
            hyp_param = get_optimal_hyperparamaters(cwd+'\\results\\tuning_summary.csv', model)
            model_ = models_loader(model)
            forecasts = model_forecast(model = model_ , series = test.values ,window_size =hyp_param[2])
            actual = test.values
        
            # Calculate the prediction intervals
            errors = actual - forecasts
            std_dev = np.std(errors)
            z_value = 1.96 # For 95% confidence intervals
            prediction_interval = z_value * std_dev
            
            
            # Plot the predicted values and their associated prediction intervals
            fig =plt.figure(figsize=(8,8))
            plt.plot(test.index,actual, label='Observed')
            plt.plot(test[hyp_param[2] - 1 :].index,forecasts, label='Out-of-sample Forecast')
            plt.fill_between(test[hyp_param[2] - 1 :].index, 
                            forecasts - prediction_interval, 
                            forecasts + prediction_interval, 
                            color='gray', 
                            alpha=0.2,
                            label='95% CI')
            # plt.title(f'Forecasted Values of {model} model with 95% Confidence Interval')
            plt.xlabel('Months')
            plt.ylabel('Inflation')
            plt.xticks(rotation= 20)
            plt.legend(loc='best')

            # Save the figure
            fig.savefig(cwd + f'\\results\\graphs\\{model}_model_forecast_test.png',dpi=300, bbox_inches='tight')
            

            for metric in metrics:
                for h in horizons:
                 loss = loss_comp(forecasts[:h], actual.reshape(-1)[len(test)-24 :len(test)-24 + h],loss = metric)
                # print(f'{metric}_{model} is:',loss)
                 results = results.append({'model': model, 'horizon': h, 'metric': metric, 'value': loss}, ignore_index=True)
    results.to_csv(cwd +f'\\results\\forecasting performence.csv') 

if __name__ == "__main__":
    main()
