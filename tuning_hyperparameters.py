import os
import pandas as pd
import itertools
from helper import *

import warnings
warnings.filterwarnings('ignore')

def main():
    cwd = os.getcwd()
    train = pd.read_csv(cwd+'\\results\\data\\training_dataset.csv',index_col='date',parse_dates=['date'])
    valid = pd.read_csv(cwd+'\\results\\data\\validation_dataset.csv',index_col='date',parse_dates=['date'])

    input_shape_num = [12,18,24]
    layer_nums      = [1,2,3,4]
    layer_units     = [16,32,64,128]


    model_name      = []
    window_size     = []
    layer_nums_itr  = []
    layer_units_itr = []
    optimal_lrs     = []
    validation_mean_squared_errors = []
    
    lr_loss_df = pd.DataFrame()
    
    for inputs in input_shape_num :
        train_dataset = windowed_dataset(series=train.values,window_size= inputs)
        valid_dataset = windowed_dataset(series=valid.values,window_size= inputs)
        train_actual =  train[inputs - 1 :]
        valid_actual =  valid[inputs - 1 :]
       
    
        for layer_num,layer_unit in itertools.product(layer_nums,layer_units):
            models  = {
                    'NN'  : NN_construction(layer_units=layer_unit,input_shape=inputs),
                    'MLP' : MLP_construction(layer_units=layer_unit,layer_num=layer_num,input_shape=inputs),
                    'simpl_RNN': simpl_RNN_construction(layer_units=layer_unit,layer_num=layer_num,input_shape=inputs),
                    'LSTM': LSTM_construction(layer_units=layer_unit,layer_num=layer_num,input_shape=inputs),
                    'BI_LSTM': BI_LSTM_construction(layer_units=layer_unit,layer_num=layer_num,input_shape=inputs),
                    'CNN': CNN_construction(layer_units=layer_unit,layer_num=layer_num,input_shape=inputs),
                    'CNN_LSTM': CNN_LSTM_construction(layer_units=layer_unit,layer_num=layer_num,input_shape=inputs)
                     }
            for model in models:   
                optimal_lr,lr_loss = tune_learning_rate(model=models[model],dataset=train_dataset,validation_data=valid_dataset)
                model_train(model = models[model],dataset=train_dataset,validation_data=valid_dataset,learning_rate=optimal_lr,patience=5)
                
                train_predicts = model_forecast(model=models[model],series=train.values,window_size=inputs)
                valid_predicts = model_forecast(model=models[model],series=valid.values,window_size=inputs)
                print(train_predicts.shape)
                print(f'Root Mean Squared Error of training_{model}_{inputs}_{layer_num}_{layer_unit} is:', loss_comp(train_predicts, train_actual.values.reshape(-1),loss='rmse'))
                print(f'Root Mean Squared Error of validation_{model}_{inputs}_{layer_num}_{layer_unit} is:', loss_comp(valid_predicts, valid_actual.values.reshape(-1),loss='rmse'))
        
                # Convert the array lr_loss to a Pandas Series
                lr_loss = pd.Series(lr_loss)
                lr_loss = pd.DataFrame({f'{model}_{layer_num}_{layer_unit}':lr_loss})
                # Add a common index to both DataFrames
                lr_loss_df.index = range(len(lr_loss_df))
                lr_loss.index = range(len(lr_loss))
                # Merge the DataFrames on the common index
                lr_loss_df = pd.concat([lr_loss_df, lr_loss], axis=1)
                
                model_name.append(model)
                window_size.append(inputs)
                layer_nums_itr.append(layer_num)
                layer_units_itr.append(layer_unit)
                optimal_lrs.append(optimal_lr)
                validation_mean_squared_errors.append(loss_comp(valid_predicts, valid_actual.values.reshape(-1),loss='rmse'))
                # traning_mean_squared_errors.append(loss_comp(train_predicts, train_actual.values.reshape(-1),loss='rmse'))
    
    # print(lr_loss_df)
    lr_loss_df.to_csv(cwd + f'\\results\\tuning_lr_loss.csv')
    
    tuning_summary = pd.DataFrame({'Model':model_name,
                                   'Input shape':inputs,
                                   'Layers':layer_nums_itr,
                                   'Units':layer_units_itr,
                                   'Learning rate':optimal_lrs,
                                   'Root Mean Squared Errors (validation)':validation_mean_squared_errors })
    # print(tuning_summary)
    tuning_summary.to_csv(cwd + f'\\results\\tuning_summary.csv')
    
if __name__ == "__main__":
    main()