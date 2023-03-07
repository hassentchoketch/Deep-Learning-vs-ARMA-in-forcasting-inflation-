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


    user_input_hyp_param = input("Please enter respectively the following optimal hyperparameters : number of layers , number of unit in each layer and optimal learning rate separated by commas: ")
    # split the input into separate values
    values = user_input_hyp_param.split(",")
    # convert the values to the appropriate data type (e.g. int, float, str)
    hyp_param = []
    for value in values:
        value = value.strip()
        if value.isdigit(): 
           hyp_param.append(int(value)) 
        elif "." in value and value.replace(".", "").isdigit(): 
           hyp_param.append(float(value)) 
        else:
           hyp_param.append(value) 

    model_lstm = LSTM_construction(layer_units=hyp_param[1],layer_num=hyp_param[0])
    history = model_train(model = model_lstm,dataset=train_dataset,validation_data=valid_dataset,learning_rate=hyp_param[2],patience=5)

    plot_training_loss(history)
    train_predicts = model_forecast(model=model_lstm,series=train.values)
    valid_predicts = model_forecast(model=model_lstm,series=valid.values)
    plot_series(train_date,(train_actual, train_predicts),path= 'Actual vs fited values (training dataset)',legend=["Actual_values", "Fited_values"],title=" Fited vs Actual values (training dataset)")
    plot_series(valid_date,(valid_actual, valid_predicts),path= 'Actual vs fited values (validation dataset)',legend=["Actual_values", "predicted_values"],title=" Fited vs predicted values (validation dataset)")

if __name__ == "__main__":
    main()