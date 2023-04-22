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

    train_dataset = windowed_dataset(series=train.values)
    valid_dataset = windowed_dataset(series=valid.values)
    
    if not os.path.exists(os.path.join(cwd+'\\results', "models")):# check if the directory exists
      os.mkdir(cwd + "\\results\\models") # create the directory if it doesn't exist
    else :
      #remove all files in the directory
      for filename in os.listdir(cwd + "\\results\\models"):
        file_path = os.path.join(cwd + "\\results\\models", filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Error deleting {file_path}: {e}')

    models  = ['LSTM', 'DNN','RNN','CNN' ]
    for model in models: 
         hyp_param = get_optimal_hyperparamaters(cwd+'\\results\\tuning_summary.csv', model)
         train_date = pd.to_datetime(train.index)[hyp_param[2] - 1 :]
         valid_date = pd.to_datetime(valid.index)[hyp_param[2] - 1 :]
         train_actual =  train[hyp_param[2] - 1 :]
         valid_actual =  valid[hyp_param[2] - 1 :]
         if model == 'NN':
            model_ = NN_construction(layer_units=hyp_param[4],layer_num=hyp_param[3],input_shape=hyp_param[2])
         elif model == 'MLP':
            model_ = MLP_construction(layer_units=hyp_param[4],layer_num=hyp_param[3],input_shape=hyp_param[2])
            
         elif model == 'simpl_RNN':
            model_ = simpl_RNN_construction(layer_units=hyp_param[4],layer_num=hyp_param[3],input_shape=hyp_param[2])
            
         elif model == 'LSTM':
            model_ = LSTM_construction(layer_units=hyp_param[4],layer_num=hyp_param[3],input_shape=hyp_param[2])
            
         elif model == 'BI_LSTM': 
            model_ = BI_LSTM_construction(layer_units=hyp_param[4],layer_num=hyp_param[3],input_shape=hyp_param[2])
            
         elif model == 'CNN': 
            model_ = CNN_construction(layer_units=hyp_param[4],layer_num=hyp_param[3],input_shape=hyp_param[2])
            
         elif model == 'CNN_LSTM': 
            model_ = CNN_LSTM_construction(layer_units=hyp_param[4],layer_num=hyp_param[3],input_shape=hyp_param[2])
            
         else : 
            raise ValueError('Invalid model name: {}'.format(model))
             
         history = model_train(model = model_,dataset=train_dataset,validation_data=valid_dataset,learning_rate=hyp_param[5],patience=5)
         plot_training_validation_loss(model=model ,history=history) # fig_name=f'{model}_training_loss.png'
         train_predicts = model_forecast(model=model_,series=train.values)
         valid_predicts = model_forecast(model=model_,series=valid.values)
         rmse = loss_comp(valid_predicts, valid_actual.values.reshape(-1),loss = 'rmse')
         mae = loss_comp(valid_predicts, valid_actual.values.reshape(-1),loss = 'mae')
         print(f'------------------------{model}--------------------')
         print(f'Validation loss (RMSE) of {model} model is:{rmse}')
         print(f'Validation loss (RMSE) of {model} model is:{mae}')
         print('---------------------------------------------------')
         models_saver(model=model_ ,name = model)
         plot_series(train_date,(train_actual, train_predicts),fig_name= f'{model} model_fit',legend=["Actual_values", "Fited_values"])#,title=f" Fited vs Actual values of {model} model (training dataset)")
         plot_series(valid_date,(valid_actual, valid_predicts),fig_name= f'{model} model_forcast_validation',legend=["Actual_values", "Forecasted_values"])#),title=f" Fited vs predicted values of {model} model (validation dataset)")
      
      
      
      
      
      
      
      # if model == 'MLP':
      #    hyp_param_mpl = get_optimal_hyperparamaters(cwd+'\\results\\tuning_summary.csv', model)
      #    train_date = pd.to_datetime(train.index)[hyp_param_mpl[2] - 1 :]
      #    valid_date = pd.to_datetime(valid.index)[hyp_param_mpl[2] - 1 :]
      #    train_actual =  train[hyp_param_mpl[2] - 1 :]
      #    valid_actual =  valid[hyp_param_mpl[2] - 1 :]
      #    model_ = MPL_construction(layer_units=hyp_param_mpl[4],layer_num=hyp_param_mpl[3],input_shape=hyp_param_mpl[2])
      #    history = model_train(model = model_,dataset=train_dataset,validation_data=valid_dataset,learning_rate=hyp_param_mpl[5],patience=5)
      #    plot_training_validation_loss(model=model ,history=history) # fig_name=f'{model}_training_loss.png'
      #    train_predicts = model_forecast(model=model_,series=train.values)
      #    valid_predicts = model_forecast(model=model_,series=valid.values)
      #    rmse = loss_comp(valid_predicts, valid_actual.values.reshape(-1),loss = 'rmse')
      #    mae = loss_comp(valid_predicts, valid_actual.values.reshape(-1),loss = 'mae')
      #    print(f'Validation loss (RMSE) of {model} model is:{rmse}')
      #    print(f'Validation loss (RMSE) of {model} model is:{mae}')
      #    models_saver(model=model_ ,name = model)
      #    plot_series(train_date,(train_actual, train_predicts),fig_name= f'{model} model_fit',legend=["Actual_values","Fited_values"])#,title=f" Fited vs Actual values of {model} model (training dataset)")
      #    plot_series(valid_date,(valid_actual, valid_predicts),fig_name= f'{model} model_forcast_validation',legend=["Actual_values","Forecasted_values"])#),title=f" Fited vs predicted values of {model} model (validation dataset)")

       
      #  if model == 'LSTM':
      #    hyp_param_lstm = get_optimal_hyperparamaters(cwd+'\\results\\tuning_summary.csv', model)
      #    model_ = LSTM_construction(layer_units=hyp_param_lstm[4],layer_num=hyp_param_lstm[3],input_shape=hyp_param_lstm[2])
      #    history = model_train(model = model_,dataset=train_dataset,validation_data=valid_dataset,learning_rate=hyp_param_lstm[5],patience=5)
      #    plot_training_validation_loss(model=model ,history=history)#fig_name=f'{model}_training_loss.png'
      #    train_predicts = model_forecast(model=model_,series=train.values)
      #    valid_predicts = model_forecast(model=model_,series=valid.values)
      #    loss = loss_comp(valid_predicts, valid_actual.values.reshape(-1),loss = 'rmse')
      #    print(f'Validation loss (RMSE) of {model} model is:{loss}')
      #    models_saver(model=model_ ,name = 'LSTM')
      #    plot_series(train_date,(train_actual, train_predicts),fig_name= f'{model} model_fit',legend=["Actual_values", "Fited_values"])#,title=f" Fited vs Actual values of {model} model (training dataset)")
      #    plot_series(valid_date,(valid_actual, valid_predicts),fig_name= f'{model} model_forcast_validation',legend=["Actual_values", "Forecasted_values"])#),title=f" Fited vs predicted values of {model} model (validation dataset)")
       
      #  elif model == 'DNN':
      #    user_input_hyp_param = input(f"Please enter respectively the following optimal hyperparameters for {model} model : number of layers , number of unit in each layer and optimal learning rate separated by commas: ")
      #     # split the input into separate values
      #    values = user_input_hyp_param.split(",")
      #     # convert the values to the appropriate data type (e.g. int, float, str)
      #    hyp_param = []
      #    for value in values:
      #       value = value.strip()
      #       if value.isdigit(): 
      #          hyp_param.append(int(value)) 
      #       elif "." in value and value.replace(".", "").isdigit(): 
      #          hyp_param.append(float(value)) 
      #       else:
      #          hyp_param.append(value) 

      #    model_ = DNN_construction(layer_units=hyp_param[1],layer_num=hyp_param[0])
      #    history = model_train(model = model_,dataset=train_dataset,validation_data=valid_dataset,learning_rate=hyp_param[2],patience=5)

      #    plot_training_validation_loss(model ,history)
      #    train_predicts = model_forecast(model=model_,series=train.values)
      #    valid_predicts = model_forecast(model=model_,series=valid.values)
         
      #    loss=loss_comp(valid_predicts, valid_actual.values.reshape(-1),loss = 'rmse')
      #    print(f'Validation loss of {model} model is : {loss}')
         
      #    models_saver(model=model_,name='DNN')
         
      #    plot_series(train_date,(train_actual, train_predicts),fig_name= f'Actual vs fited values of {model} model (training dataset)',legend=["Actual_values", "Fited_values"],title=f" Fited vs Actual values of {model} model (training dataset)")
      #    plot_series(valid_date,(valid_actual, valid_predicts),fig_name= f'Actual vs predicted values of {model} model (validation dataset)',legend=["Actual_values", "predicted_values"],title=f" Fited vs predicted values of {model} model (validation dataset)")
      #  elif model == 'RNN':
      #    user_input_hyp_param = input(f"Please enter respectively the following optimal hyperparameters for {model} model : number of layers , number of unit in each layer and optimal learning rate separated by commas: ")
      #     # split the input into separate values
      #    values = user_input_hyp_param.split(",")
      #     # convert the values to the appropriate data type (e.g. int, float, str)
      #    hyp_param = []
      #    for value in values:
      #       value = value.strip()
      #       if value.isdigit(): 
      #          hyp_param.append(int(value)) 
      #       elif "." in value and value.replace(".", "").isdigit(): 
      #          hyp_param.append(float(value)) 
      #       else:
      #          hyp_param.append(value) 

      #    model_ = RNN_construction(layer_units=hyp_param[1],layer_num=hyp_param[0])
      #    history = model_train(model = model_,dataset=train_dataset,validation_data=valid_dataset,learning_rate=hyp_param[2],patience=5)

      #    plot_training_validation_loss(model ,history)
      #    train_predicts = model_forecast(model=model_,series=train.values)
      #    valid_predicts = model_forecast(model=model_,series=valid.values)
         
      #    loss=loss_comp(valid_predicts, valid_actual.values.reshape(-1),loss = 'rmse')
      #    print(f'Validation loss of {model} model is : {loss}')
         
      #    models_saver(model=model_,name = 'RNN')
         
      #    plot_series(train_date,(train_actual, train_predicts),fig_name= f'Actual vs fited values of {model} model (training dataset)',legend=["Actual_values", "Fited_values"],title=f" Fited vs Actual values of {model} model (training dataset)")
      #    plot_series(valid_date,(valid_actual, valid_predicts),fig_name= f'Actual vs predicted values of {model} model (validation dataset)',legend=["Actual_values", "predicted_values"],title=f" Fited vs predicted values of {model} model (validation dataset)")
      #  elif model == 'CNN':
      #    user_input_hyp_param = input(f"Please enter respectively the following optimal hyperparameters for {model} model : number of layers , number of unit in each layer and optimal learning rate separated by commas: ")
      #     # split the input into separate values
      #    values = user_input_hyp_param.split(",")
      #     # convert the values to the appropriate data type (e.g. int, float, str)
      #    hyp_param = []
      #    for value in values:
      #       value = value.strip()
      #       if value.isdigit(): 
      #          hyp_param.append(int(value)) 
      #       elif "." in value and value.replace(".", "").isdigit(): 
      #          hyp_param.append(float(value)) 
      #       else:
      #          hyp_param.append(value) 

      #    model_ = CNN_construction(layer_units=hyp_param[1],layer_num=hyp_param[0])
      #    history = model_train(model = model_,dataset=train_dataset,validation_data=valid_dataset,learning_rate=hyp_param[2],patience=5)

      #    plot_training_validation_loss(model ,history)
      #    train_predicts = model_forecast(model=model_,series=train.values)
      #    valid_predicts = model_forecast(model=model_,series=valid.values)
         
      #    loss=loss_comp(valid_predicts, valid_actual.values.reshape(-1),loss = 'rmse')
      #    print(f'Validation loss of {model} model is : {loss}')
         
      #    models_saver(model=model_,name = 'CNN')
      #    plot_series(train_date,(train_actual, train_predicts),fig_name= f'Actual vs fited values of {model} model (training dataset)',legend=["Actual_values", "Fited_values"],title=f" Fited vs Actual values of {model} model (training dataset)")
      #    plot_series(valid_date,(valid_actual, valid_predicts),fig_name= f'Actual vs predicted values of {model} model (validation dataset)',legend=["Actual_values", "predicted_values"],title=f" Predicted vs Actual values of {model} model (validation dataset)")
      #  else:
      #   raise ValueError('Invalid model name: {}'.format(model))
if __name__ == "__main__":
    main()