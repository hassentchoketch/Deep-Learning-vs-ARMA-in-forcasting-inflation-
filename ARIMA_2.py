import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

from helper import *
import os


class Arima_Class:
    def __init__(self, arima_para, seasonal_para,start_period,end_period):
        # Define the p, d and q parameters in Arima(p,d,q)(P,D,Q) models
        p = arima_para['p']
        d = arima_para['d']
        q = arima_para['q']
        # Generate all different combinations of p, q and q triplets
        self.pdq = list(itertools.product(p, d, q))
        # Generate all different combinations of seasonal p, q and q triplets
        # self.seasonal_pdq = [(x[0], x[1], x[2], seasonal_para)
                            #  for x in list(itertools.product(p, d, q))]
        self.start_period = start_period
        self.end_period = end_period
                   
    def tuning_parameters(self,ts):
        warnings.filterwarnings("ignore")
        self.results_list = []
        for param in self.pdq:
            # for param_seasonal in self.seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(ts[self.start_period:self.end_period],
                                                    order=param,
                                                    # seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    
                    results = mod.fit(disp=False)

                    print('ARIMA{} with AIC={} and BIC={}'.format(param,results.aic, results.bic))
                                                            #    param_seasonal, 
                    self.results_list.append([param, results.aic, results.bic])
                                        #  param_seasonal, results.aic, results.bic])
                except:
                    continue
        
    def fit(self, ts ,criteria ):
        self.criteria = criteria
        results_list = np.array(self.results_list)
        lowest_AIC = np.argmin(results_list[:, 1])
        lowest_BIC = np.argmin(results_list[:, 2])
        
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        if self.criteria == 'AIC':
           print('ARIMA{} with lowest_AIC:{}'.format(
            results_list[lowest_AIC, 0], results_list[lowest_AIC, 1]))
           print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
           mod = sm.tsa.statespace.SARIMAX(ts[self.start_period:self.end_period],
                                        order=results_list[lowest_AIC, 0],
                                        # seasonal_order=results_list[lowest_AIC, 1],
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
           
        else:
           print('ARIMA{} with lowest_BIC:{}'.format(
            results_list[lowest_BIC, 0], results_list[lowest_BIC, 2]))
           print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
           mod = sm.tsa.statespace.SARIMAX(ts[self.start_period:self.end_period],
                                        order=results_list[lowest_BIC, 0],
                                        # seasonal_order=results_list[lowest_BIC, 1],
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
           
        self.final_result = mod.fit(disp=False)
        # print(f'Final ARIMA-{self.criteria} model summary:')
        # print(self.final_result.summary().tables[1])
        # print(f'Final ARIMA-{self.criteria} model diagnostics:')
        self.final_result.plot_diagnostics(figsize=(15, 12))
        plt.tight_layout()
        plt.savefig(cwd + f'\\results\\graphs\\ARIMA-{self.criteria}_model_diagnostics.png', dpi=300)
        plt.show()

    def pred(self, ts, plot_start, pred_start, dynamic, ts_label):

        pred_dynamic = self.final_result.get_prediction(
            start=pd.to_datetime(pred_start), dynamic=dynamic, full_results=True)
        pred_dynamic_ci = pred_dynamic.conf_int()
        ax = ts[plot_start:].plot(label='observed', figsize=(15, 10))

        if dynamic == False:
            pred_dynamic.predicted_mean.plot(
                label='One-step ahead Forecast', ax=ax)
        else:
            pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

        ax.fill_between(pred_dynamic_ci.index,
                        pred_dynamic_ci.iloc[:, 0],
                        pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
        ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(plot_start), ts.index[-1],
                         alpha=.1, zorder=-1)
        ax.set_xlabel('Time')
        ax.set_ylabel(ts_label)
        plt.legend()
        plt.tight_layout()
        if dynamic == False:
            plt.savefig(ts_label + '_one_step_pred.png', dpi=300)
        else:
            plt.savefig(ts_label + '_dynamic_pred.png', dpi=300)
        plt.show()

    def forcast(self, ts, plot_start, n_steps, ts_label):
        # Get forecast n_steps ahead in future
        pred_uc = self.final_result.get_forecast(steps=n_steps)

        # Get confidence intervals of forecasts
        pred_ci = pred_uc.conf_int()
        # plot
        # Create a new figure and an AxesSubplot object
        fig, ax = plt.subplots(figsize=(8, 8))
        # ax = ts[plot_start:].plot(label={'CPI(%)':'Observed'}, figsize=(15, 10))
        # Plot the observed data
        ax.plot(ts[plot_start:].index, ts[plot_start:], label='Observed')
        
        # Plot the predicted mean
        pred_uc.predicted_mean.plot(ax=ax, label='Out-of-sample Forecast')
        # Plot the confidence interval
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='gray', alpha=.20,
                        label='CI')
        # Add labels and legend
        ax.set_xlabel('Months')
        ax.set_ylabel(ts_label)
        # Save the figure and show the plot
        plt.tight_layout()
        plt.legend()
        plt.savefig(cwd + f'\\results\\graphs\\ARIMA-{self.criteria}_model_forcast.png', dpi=300,bbox_inches='tight')
        
        plt.show()
        return pred_uc.predicted_mean
    
    def test_performence(self, ts, forcast):
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # results =[]
        for metrics in ['RMSE', 'MAE']:
          for h in [1,3,6,12,24] :
              if metrics == 'RMSE':
                rmse = np.sqrt(mean_squared_error(ts[len(ts)-24:len(ts)-24 + h], forcast[:h]))
                print('ARIMA-{} model forcast results : metrics {} horizon {} loss {}'.format(self.criteria,metrics,h,rmse))
              elif metrics == 'MAE': 
                mae = mean_absolute_error(ts[len(ts)-24:len(ts)-24 + h], forcast[:h])
                print('ARIMA-{} model forcast results : metrics {} horizon {} loss {}'.format(self.criteria,metrics,h,mae))
                # results.append[h,metrics,rmse,mae]
                
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
       
#    -------------------------------------------------------------------------

cwd = os.getcwd()
config_plot()
inf = load_transform_data(data_file = 'DZ_Consumption_price_index.csv')
ts_label = 'Inflation rate'

# "Grid search" of seasonal ARIMA model.

arima_para = {'p':range(1,13),'d':range(1,2),'q':range(1,13)}    
    
# the seasonal periodicy is  12 month
seasonal_para = 12

test_steps = 24


    
arima = Arima_Class(arima_para, seasonal_para,start_period= inf.index[0] , end_period= inf.index[-test_steps] )
arima.tuning_parameters(inf)

for criteria in ['AIC','BIC']:
    # fitting the model
    arima.fit(inf,criteria=criteria)

    # Forecasts to unseen future data
    forecast = arima.forcast(inf,plot_start= '2019', n_steps= 24 , ts_label =ts_label )
    
    # forecasting performence on test set 
    arima.test_performence(inf,forecast)