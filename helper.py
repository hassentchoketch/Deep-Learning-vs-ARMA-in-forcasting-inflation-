import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import tensorflow as tf 

cwd = os.getcwd()

def change_perc(series,lenth):
  diff_data=[]
  for i in range(lenth , len(series)):
    value = (series[i] - series[i-lenth])/series[i-lenth] * 100
    diff_data.append(value)
  return pd.Series(diff_data,index=series.index[12:])

def load_transform_data(path= None, series= 'CPI', lenth= 12, stat_date= '1998'):
  
    df = pd.read_csv(cwd+f'\\results\\data\\{path}')
    df[f'{series}(%)'] = change_perc(df[series],lenth=lenth)
    df.set_index('date',inplace= True)
    df.index = pd.to_datetime(df.index)
    var = df.loc[stat_date:,f'{series}(%)'].reset_index()
    var.set_index('date',inplace= True)
    return var

def plot_description(time_series=None, title = None, path=None ,window_size = 0.05,test_split_ratio=0.1,valid_split_ratio=0.2, if_split=False ):

    """
    time_series: the time series which will be plotted
    name: used for later label and title.
    effect: return the plot containing original time series and rolling mean and rolling std.
    """
    fig = plt.gcf()
    fig.set_size_inches(16, 6)
    plt.style.use('seaborn')
    if if_split:
        train = time_series[0 : round(len(time_series) * (1-test_split_ratio))]
        test  = time_series[round(len(time_series) * (1-test_split_ratio)) :]
        valid = train[round(len(train) * (1-valid_split_ratio)) :] 
        train = train[0 : round(len(train) * (1-valid_split_ratio))]
        plt.plot(train, label="train")
        plt.plot(test, label="test")
        plt.plot(valid, label="valid")
        plt.plot( time_series.rolling(int(window_size * len(train))).mean(), "--", label="Rolling mean")
        plt.plot(time_series.rolling(int(window_size * len(train))).std(), ":", label="Rolling Std")
    else:
        plt.plot(time_series, label="Series")
        plt.plot(time_series.rolling(int(window_size * len(time_series))).mean(),"--",label="Rolling mean",)
        plt.plot(time_series.rolling(int(window_size * len(time_series))).std(),":",label="Rolling Std",)
    plt.title("Overview Description Plot of " + title.replace("_", " "),fontsize=20,fontweight='medium')
    plt.legend(loc="best")
    plt.savefig(cwd+f'\\results\\graphs\\{path}')
    # plt.show()
    plt.close()

def plot_decomposition(time_series=None, var_name = None, path=None):

    """
    time_series: the time series which will be plotted
    name: used for later label and title.
    effect: return the plot containing original time series and rolling mean and rolling std.
    """
    
    
    plt.style.use('seaborn')
    decomposition = sm.tsa.seasonal_decompose(time_series, model='additive')
    fig =decomposition.plot()
    fig.suptitle(f"{var_name} decomposition (Seasonal and trend componts)",fontsize=20,fontweight='medium')
    # plt.legend(loc="best")
    plt.savefig(cwd + f'\\results\\graphs\\{path}')
    # plt.show()
    plt.close()

def plot_heatmap(time_series=None,name ='CPI',path= None):
    
    year = time_series.index.year
    month = time_series.index.month_name()
    frame ={'inf' : time_series[name] ,'year': year, 'month':month }
    df =pd.DataFrame(frame)
    df_pivot = df.pivot_table(index="month", columns="year",values='inf',fill_value=0)
    
    fig = plt.figure(figsize=(16,8))
    ax = plt.subplot()
    sns.heatmap(df_pivot,cmap = 'viridis',annot=True, linewidth=.5,ax=ax)#cmap='Blues'
    plt.title('CPI Inflation Heat Map(Percent Change ,YoY)',fontsize=20,fontweight='medium')
    plt.savefig(f'results\graphs\{path}', dpi=300)
    # plt.show()
    plt.close()

def data_split(time_series=None,test_split_ratio=0.1,valid_split_ratio = 0.2):
    if test_split_ratio > 0.2 or test_split_ratio<= 0:
        test_split_ratio = 0.2
    if valid_split_ratio > 0.3 or valid_split_ratio<= 0:
        test_split_ratio = 0.3

    train = time_series[0 : round(len(time_series) * (1-test_split_ratio))]
    test  = time_series[round(len(time_series) * (1-test_split_ratio))-12 :]
    valid = train[round(len(train) * (1-valid_split_ratio)) :]
    train = train[0 : round(len(train) * (1-valid_split_ratio))]
    
    
    train.to_csv(cwd+'\\results\\data\\training_dataset.csv')
    test.to_csv(cwd+ '\\results\\data\\testing_dataset.csv')
    valid.to_csv(cwd+ '\\results\data\\validation_dataset.csv')

def windowed_dataset(series=None, window_size=12, batch_size=30, shuffle_buffer=100):

    """Generates dataset windows
    Args:
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the feature
      batch_size (int) - the batch size
      shuffle_buffer(int) - buffer size to use for the shuffle method

     Returns:
      dataset (TF Dataset) - TF Dataset containing time windows
    """
    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Create tuples with features and labels
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    # Shuffle the windows
    dataset = dataset.shuffle(shuffle_buffer)

    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset

def LSTM_construction(input_shape=12,layer_num=2,layer_units=64,output_units=1,activation='relu'):

    """
    input_layer : number of denses in input layer.
    layer_units: number of denses in hideen layers.
    layer_num : number of hidden layers.
    activation_func: activation function.
    input_shape : number of time steps in the input sequence (window_size) 
    Return :LSTM model
    """
    if layer_num == 1:
        LSTM = tf.keras.models.Sequential(
            [
                tf.keras.layers.Lambda(
                    lambda x: tf.expand_dims(x, axis=-1), input_shape=[input_shape]
                ),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        layer_units, return_sequences=True, activation=activation
                    )
                ),
                tf.keras.layers.Dense(output_units),
                # tf.keras.layers.Lambda(lambda x: x * 100.0),
            ]
        )
    if layer_num == 2:
        LSTM = tf.keras.models.Sequential(
            [
                tf.keras.layers.Lambda(
                    lambda x: tf.expand_dims(x, axis=-1), input_shape=[input_shape]
                ),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        layer_units, return_sequences=True, activation=activation
                    )
                ),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(layer_units, activation=activation)
                ),
                tf.keras.layers.Dense(output_units),
                # tf.keras.layers.Lambda(lambda x: x * 100.0),
            ]
        )
    if layer_num == 3:
        LSTM = tf.keras.models.Sequential(
            [
                tf.keras.layers.Lambda(
                    lambda x: tf.expand_dims(x, axis=-1), input_shape=[input_shape]
                ),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        layer_units, return_sequences=True, activation=activation
                    )
                ),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        layer_units, return_sequences=True, activation=activation)
                ),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(layer_units, activation=activation)
                ),
                tf.keras.layers.Dense(output_units),
                # tf.keras.layers.Lambda(lambda x: x * 100.0),
            ]
        )
    return LSTM

def tune_learning_rate(model=None,dataset= None,validation_data=None,path=None, patience = 5 ,loss='mse', epochs= 100, title='Tune_learning_rate',momentum=0.9,plot = False):

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-5 * 10 ** (epoch / 20)
    )
    optimizer = tf.keras.optimizers.SGD(momentum)
    model.compile(loss=loss, optimizer=optimizer)
    # Define early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    history = model.fit(dataset, epochs=epochs, validation_data=validation_data, callbacks=[lr_schedule, early_stopping])
    lrs = 1e-5 * (10 ** (np.arange(epochs) / 20))
    if plot:
      # Plot losses vs learning rates
      epochs = len(history.history['loss'])
      plt.figure(figsize=(10, 4))
      plt.grid(True)
      plt.semilogx(lrs, history.history["loss"])
      plt.tick_params("both", length=10, width=1, which="both")
      plt.axis([1e-5, 1e-1, 0, 20])
      plt.title(title)
      plt.savefig(cwd + f'\\results\\graphs\\{path}')
      plt.close()
    
    # Select optimal learning rate
    optimal_lr = lrs[np.argmin(history.history['val_loss'])]
    print('Optimal learning rate:', optimal_lr)
    return optimal_lr

def model_train(model=None,dataset= None,validation_data=None,learning_rate=None,patience=5,loss='mse',metrics='mae', epochs= 100,momentum=0.9):
    callback = tf.keras.callbacks.EarlyStopping(patience=patience)
    # learning_rate = float(input("Pleas enter the optimal lr:"))
    optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=momentum)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
    history = model.fit(dataset, epochs=epochs,validation_data = validation_data,callbacks=[callback])
    return history

def loss_comp(y_pred, y_truth, loss):

    """
    y_pred: predicted values.
    y_truth: ground truth.
    loss_func: the function used to calculate loss.
    return loss value.
    """
    if loss == "mse":
        return tf.keras.metrics.mean_squared_error(y_pred, y_truth).numpy()
        # mean_squared_error(y_truth, y_pred)
    elif loss == "mae":
        return tf.keras.metrics.mean_absolute_error(y_pred, y_truth).numpy()
        # r2_score(y_truth, y_pred)
    else:
        if loss != "rmse":
            # logger.warning(
            print("Error:The loss functin is illegal. Turn to default loss function: rmse" )
    return tf.keras.metrics.RootMeanSquaredError(y_pred, y_truth).numpy()

def plot_series(x,y,path,format="-",start=0,end=None,title=None,xlabel=None,ylabel=None,legend=None,loss=None):
    """
    Visualizes time series data

    Args:
      x (array of int) - contains values for the x-axis
      y (array of int or tuple of arrays) - contains the values for the y-axis
      format (string) - line style when plotting the graph
      start (int) - first time step to plot
      end (int) - last time step to plot
      title (string) - title of the plot
      xlabel (string) - label for the x-axis
      ylabel (string) - label for the y-axis
      legend (list of strings) - legend for the plot
      name (string) - name for the graph
    """

    # Setup dimensions of the graph figure
    plt.figure(figsize=(10, 6))
    # Check if there are more than two series to plot
    if type(y) is tuple:
        # Loop over the y elements
        for y_curr in y:
            # Plot the x and current y values
            plt.plot(x[start:end], y_curr[start:end], format)
    else:
        # Plot the x and y values
        plt.plot(x[start:end], y[start:end], format)
    # Label the x-axis
    plt.xlabel(xlabel)
    # Label the y-axis
    plt.ylabel(ylabel)
    # Set the legend
    if legend:
        plt.legend(legend, loc="best")
    # Set the title
    plt.title(title)
    # Overlay a grid on the graph
    plt.grid(True)
    plt.text(0.95, 0.01, " ** Loss value = : {:0.4f}".format(loss), verticalalignment='bottom', horizontalalignment='right',color='green',fontsize=12)

    plt.savefig(cwd + f'/results/graphs/{path}')
    # Draw the graph on screen
    plt.close()

def plot_training_loss(history ,path ='LSTM training_loss.png'):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plot_series(epochs, (loss,val_loss), path, title="training_loss ")
    
def loss_comp(y_pred, y_truth, loss= 'mse'):
    """
    y_pred: predicted values.
    y_truth: ground truth.
    loss_func: the function used to calculate loss.
    return loss value.
    """
    if loss == "mse":
        return tf.keras.metrics.mean_squared_error(y_pred, y_truth).numpy()
        # mean_squared_error(y_truth, y_pred)
    elif loss == "mae":
        return tf.keras.metrics.mean_absolute_error(y_pred, y_truth).numpy()
        # r2_score(y_truth, y_pred)
    else:
        if loss != "rmse":
            print(
                "The loss functin is illegal. Turn to default loss function: mse"
            )
        return tf.keras.metrics.RootMeanSquaredError(y_pred, y_truth).numpy
        
def model_forecast(model, series, window_size=12, batch_size=30):

    """Uses an input model to generate predictions on data windows

    Args:
      model (TF Keras Model) - model that accepts data windows
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the window
      batch_size (int) - the batch size

    Returns:
      forecast (numpy array) - array containing predictions
    """

    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda w: w.batch(window_size))

    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)

    # Get predictions on the entire dataset
    forecast = model.predict(dataset)

    return forecast.squeeze()

def models_saver( model,loss=None):
  if not os.path.exists(os.path.join(cwd+'\\results', "models")):
    os.mkdir(cwd + "\\results\\models")
  model.save(cwd + "\\results\\models\\LSTM_model_with_val_loss_{:0.4f}.h5".format(loss))

def models_loader (): 
  model_path = input("Pleas enter the model file name:")
  model = tf.keras.models.load_model(cwd + f'\\results\\models\\{model_path}',compile=False )
  return model