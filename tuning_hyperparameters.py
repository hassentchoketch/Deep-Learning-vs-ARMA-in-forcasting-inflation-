import os
import pandas as pd
import itertools
from helper import *

def main():
    cwd = os.getcwd()
    train = pd.read_csv(cwd+'\\results\\data\\training_dataset.csv',index_col='date')
    valid = pd.read_csv(cwd+'\\results\\data\\validation_dataset.csv',index_col='date')

    train_dataset = windowed_dataset(series=train.values)
    valid_dataset = windowed_dataset(series=valid.values)
    
    train_date = pd.to_datetime(train.index)[12 - 1 :]
    valid_date = pd.to_datetime(valid.index)[12 - 1 :]
    train_actual =  train[12 - 1 :]
    valid_actual =  valid[12 - 1 :]
    
    layer_nums = [2]
    layer_units = [64, 128, 256, 512]
    
    traning_mean_squared_errors = []
    validation_mean_squared_errors = []
    layer_units_itr = []
    layer_nums_itr = []
    optimal_lrs = []

    for layer_num,layer_unit in itertools.product(layer_nums,layer_units):

        model_lstm = LSTM_construction(layer_units=layer_unit,layer_num=layer_num)
        optimal_lr = tune_learning_rate(model=model_lstm,dataset=train_dataset,validation_data=valid_dataset, path = 'LSTM_tune_learning_rate.png',patience=5)
        
        model_lstm = LSTM_construction(layer_units=layer_unit,layer_num=layer_num)
        history = model_train(model = model_lstm,dataset=train_dataset,validation_data=valid_dataset,learning_rate=optimal_lr,patience=5)
          
        train_predicts = model_forecast(model=model_lstm,series=train.values)
        valid_predicts = model_forecast(model=model_lstm,series=valid.values)
  
        print('Mean Squared Error of training is:', loss_comp(train_predicts, train_actual.values.reshape(-1)))
        print('Mean Squared Error of validation is:', loss_comp(valid_predicts, valid_actual.values.reshape(-1)))
  
        models_saver(model=model_lstm,loss=loss_comp(valid_predicts, valid_actual.values.reshape(-1)))

        layer_nums_itr.append(layer_num)
        layer_units_itr.append(layer_unit)
        optimal_lrs.append(optimal_lr)
        validation_mean_squared_errors.append(loss_comp(valid_predicts, valid_actual.values.reshape(-1)))
        traning_mean_squared_errors.append(loss_comp(train_predicts, train_actual.values.reshape(-1)))

    tuning_summary = pd.DataFrame({'Layers':layer_nums_itr,'Units':layer_units_itr,'learning rate':optimal_lrs,'Mean Squared Errors (validation)':validation_mean_squared_errors,'Mean Squared Errors (training)':traning_mean_squared_errors})
    tuning_summary.to_csv(cwd + '/results/tuning_summary.csv')

if __name__ == "__main__":
    main()