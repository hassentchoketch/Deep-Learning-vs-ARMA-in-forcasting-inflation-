import os
import pandas as pd
import itertools
from helper import *

import warnings
warnings.filterwarnings('ignore')

def main():
    config_plot()
    cwd = os.getcwd()
    train = pd.read_csv(cwd+'\\results\\data\\training_dataset.csv',index_col='date')
    valid = pd.read_csv(cwd+'\\results\\data\\validation_dataset.csv',index_col='date')
    
    if not os.path.exists(os.path.join(cwd+'\\results', "models")):# check if the directory exists
      os.mkdir(cwd + "\\results\\models") # create the directory if it doesn't exist
   #  else :
   #    #remove all files in the directory
   #    for filename in os.listdir(cwd + "\\results\\models"):
   #      file_path = os.path.join(cwd + "\\results\\models", filename)
   #      try:
   #          if os.path.isfile(file_path) or os.path.islink(file_path):
   #              os.unlink(file_path)
   #      except Exception as e:
   #          print(f'Error deleting {file_path}: {e}')

    models  = ['NN','MLP','simpl_RNN','LSTM','BI_LSTM','CNN' ]
   
    for model in models: 
         hyp_param = get_optimal_hyperparamaters(cwd+'\\results\\tuning_summary.csv', model)
         train_date = pd.to_datetime(train.index)[hyp_param[2] - 1 :]
         valid_date = pd.to_datetime(valid.index)[hyp_param[2] - 1 :]
         train_actual =  train[hyp_param[2] - 1 :]
         valid_actual =  valid[hyp_param[2] - 1 :]
         if model == 'NN':
            train_dataset = windowed_dataset(series=train.values,window_size=hyp_param[2])
            valid_dataset = windowed_dataset(series=valid.values,window_size=hyp_param[2])
            model_ = NN_construction(layer_units=hyp_param[4],input_shape=hyp_param[2])
         elif model == 'MLP':
            train_dataset = windowed_dataset(series=train.values,window_size=hyp_param[2])
            valid_dataset = windowed_dataset(series=valid.values,window_size=hyp_param[2])
            model_ = MLP_construction(layer_units=hyp_param[4],layer_num=hyp_param[3],input_shape=hyp_param[2])
            
         elif model == 'simpl_RNN':
            train_dataset = windowed_dataset(series=train.values,window_size=hyp_param[2])
            valid_dataset = windowed_dataset(series=valid.values,window_size=hyp_param[2])
            model_ = simpl_RNN_construction(layer_units=hyp_param[4],layer_num=hyp_param[3],input_shape=hyp_param[2])
            
         elif model == 'LSTM':
            train_dataset = windowed_dataset(series=train.values,window_size=hyp_param[2])
            valid_dataset = windowed_dataset(series=valid.values,window_size=hyp_param[2])
            model_ = LSTM_construction(layer_units=hyp_param[4],layer_num=hyp_param[3],input_shape=hyp_param[2])
            
         elif model == 'BI_LSTM': 
            train_dataset = windowed_dataset(series=train.values,window_size=hyp_param[2])
            valid_dataset = windowed_dataset(series=valid.values,window_size=hyp_param[2])
            model_ = BI_LSTM_construction(layer_units=hyp_param[4],layer_num=hyp_param[3],input_shape=hyp_param[2])
            
         elif model == 'CNN': 
            train_dataset = windowed_dataset(series=train.values,window_size=hyp_param[2])
            valid_dataset = windowed_dataset(series=valid.values,window_size=hyp_param[2])
            model_ = CNN_construction(layer_units=hyp_param[4],layer_num=hyp_param[3],input_shape=hyp_param[2])
            
         # elif model == 'CNN_LSTM': 
         #    model_ = CNN_LSTM_construction(layer_units=hyp_param[4],layer_num=hyp_param[3],input_shape=hyp_param[2])
            
         else : 
            raise ValueError('Invalid model name: {}'.format(model))
             
         history = model_train(model = model_,dataset=train_dataset,validation_data=valid_dataset,learning_rate=hyp_param[5],patience=15)
         plot_training_validation_loss(model=model ,history=history) # fig_name=f'{model}_training_loss.png'
         train_predicts = model_forecast(model=model_,series=train.values,window_size =hyp_param[2] )
         valid_predicts = model_forecast(model=model_,series=valid.values,window_size =hyp_param[2])
         rmse = loss_comp(valid_predicts, valid_actual.values.reshape(-1),loss = 'rmse')
         mae = loss_comp(valid_predicts, valid_actual.values.reshape(-1),loss = 'mae')
         print(f'------------------------{model}--------------------')
         print(f'The bast paramaters are : input shape = {hyp_param[2]}, numberof layers = {hyp_param[3]},number of units = {hyp_param[4]}, optimal learning rate  = {hyp_param[5]}')
         print(f'Validation loss (RMSE) of {model} model is:{rmse}')
         print(f'Validation loss (RMSE) of {model} model is:{mae}')
         print('---------------------------------------------------')
         models_saver(model=model_ ,name = model)
         plot_series(train_date,(train_actual, train_predicts),fig_name= f'{model} model_fit',legend=["Actual_values", "Fited_values"])#,title=f" Fited vs Actual values of {model} model (training dataset)")
         plot_series(valid_date,(valid_actual, valid_predicts),fig_name= f'{model} model_forcast_validation',legend=["Actual_values", "Forecasted_values"])#),title=f" Fited vs predicted values of {model} model (validation dataset)")
      
if __name__ == "__main__":
    main()