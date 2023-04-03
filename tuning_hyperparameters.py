import os
import pandas as pd
import itertools
from helper import *

def main():
    cwd = os.getcwd()
    train = pd.read_csv(cwd+'\\results\\data\\training_dataset.csv',index_col='date',parse_dates=['date'])
    valid = pd.read_csv(cwd+'\\results\\data\\validation_dataset.csv',index_col='date',parse_dates=['date'])

    train_dataset = windowed_dataset(series=train.values)
    valid_dataset = windowed_dataset(series=valid.values)
      
    train_actual =  train[12 - 1 :]
    valid_actual =  valid[12 - 1 :]
    
    layer_nums = [2,3]
    layer_units = [32]#, 64, 128, 256]


    model_name=[]
    layer_nums_itr = []
    layer_units_itr = []
    optimal_lrs = []
    traning_mean_squared_errors = []
    validation_mean_squared_errors = []
    
    
    lr_loss_df = pd.DataFrame()

    for layer_num,layer_unit in itertools.product(layer_nums,layer_units):
        models  = {'LSTM' : LSTM_construction(layer_units=layer_unit,layer_num=layer_num),
                   'DNN'  : DNN_construction(layer_units=layer_unit,layer_num=layer_num),
                   'RNN'  : RNN_construction(layer_units=layer_unit,layer_num=layer_num),
                   'CNN'  : CNN_construction(layer_units=layer_unit,layer_num=layer_num)
                    }
        for model in models:   
            optimal_lr,lr_loss = tune_learning_rate(model=models[model],dataset=train_dataset,validation_data=valid_dataset)
            model_train(model = models[model],dataset=train_dataset,validation_data=valid_dataset,learning_rate=optimal_lr,patience=5)
            
            train_predicts = model_forecast(model=models[model],series=train.values)
            valid_predicts = model_forecast(model=models[model],series=valid.values)
    
            print(f'Mean Squared Error of training_{model}_{layer_num}_{layer_unit} is:', loss_comp(train_predicts, train_actual.values.reshape(-1)))
            print(f'Mean Squared Error of validation_{model}_{layer_num}_{layer_unit} is:', loss_comp(valid_predicts, valid_actual.values.reshape(-1)))
    
            # models_saver(model=models[model],loss=loss_comp(valid_predicts, valid_actual.values.reshape(-1)))
            
            # Convert the array lr_loss to a Pandas Series
            lr_loss = pd.Series(lr_loss )
            lr_loss = pd.DataFrame({f'{model}_{layer_num}_{layer_unit}':lr_loss})
            # Add a common index to both DataFrames
            lr_loss_df.index = range(len(lr_loss_df))
            lr_loss.index = range(len(lr_loss))
            # Merge the DataFrames on the common index
            lr_loss_df = pd.concat([lr_loss_df, lr_loss], axis=1)
            
            model_name.append(model)
            layer_nums_itr.append(layer_num)
            layer_units_itr.append(layer_unit)
            optimal_lrs.append(optimal_lr)
            validation_mean_squared_errors.append(loss_comp(valid_predicts, valid_actual.values.reshape(-1)))
            traning_mean_squared_errors.append(loss_comp(train_predicts, train_actual.values.reshape(-1)))
    
    # lr_loss_df=lr_loss_df.T
    print(lr_loss_df)
    lr_loss_df.to_csv(cwd + f'\\results\\tuning_lr_loss.csv')
    
    tuning_summary = pd.DataFrame({'Model':model_name,'Layers':layer_nums_itr,'Units':layer_units_itr,'learning rate':optimal_lrs,'Mean Squared Errors (validation)':validation_mean_squared_errors,'Mean Squared Errors (training)':traning_mean_squared_errors})
    print(tuning_summary)
    tuning_summary.to_csv(cwd + f'\\results\\tuning_summary.csv')
    
if __name__ == "__main__":
    main()