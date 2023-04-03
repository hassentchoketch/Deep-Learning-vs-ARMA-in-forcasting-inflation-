import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import tensorflow as tf 

cwd = os.getcwd()

def get_percentage_change(series: pd.Series,length:int) -> pd.Series:
    """
    Returns the percentage change between values in the given series.

    Args:
         series : A Pandas Series object containing the values to compute the percentage of.  
         length  (int) :  An integer representing the number of periods to use when calculating the percentage change.
    Returns :
         A Pandas Series object containing the percentage change between values in input series.

     Raises:
         ValueError: If the input series is not a Pandas Series object or if input length is greater than or equal to the langth of the input series.          
    """
    # check that the input series is a Pandas Series object 
    if not isinstance(series , pd.Series):
      raise ValueError('Input series must be a Pandas Series object')
   
    # Check that the input series has a DatetimeIndex
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
        # raise ValueError("Input series must have a DatetimeIndex")
   
    # Check that the input length is valid
    if length >= len(series):
        raise ValueError("Input length must be less than the length of the input series")
  
    new_series =[(series[i] - series[i-length])/series[i-length] * 100 for i in range(length , len(series)) ]
    return pd.Series(new_series,index=series.index[length:])

def load_transform_data(path= None, series= 'CPI(%)', lenth= 12, stat_date= '1998'):
  
    df = pd.read_csv(cwd+f'\\data\\{path}')
    df[f'{series}'] = get_percentage_change(df[series],lenth=lenth)
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
    """Split a time series data into training, testing, and validation sets and save them as CSV files.

    Args:
        time_series (pandas DataFrame): A pandas DataFrame containing the time series data to split.
            The time series data should be in the form of a single column with the datetime as the index.
        test_split_ratio (float): The ratio of the data to use for testing. Default value is 0.1 (10%).
            Must be a value between 0 and 0.2.
        valid_split_ratio (float): The ratio of the training data to use for validation. Default value is 0.2 (20%).
            Must be a value between 0 and 0.3.

    Returns:
        tuple: A tuple containing three pandas DataFrames, in the order of (train, test, valid).
            train: The training set.
            test: The testing set.
            valid: The validation set.

    """

    if test_split_ratio > 0.2 or test_split_ratio<= 0:
        test_split_ratio = 0.2
    if valid_split_ratio > 0.3 or valid_split_ratio<= 0:
        test_split_ratio = 0.3

    train = time_series[0 : round(len(time_series) * (1-test_split_ratio))]
    test  = time_series[round(len(time_series) * (1-test_split_ratio))-12 :]
    valid = train[round(len(train) * (1-valid_split_ratio)) :]
    train = train[0 : round(len(train) * (1-valid_split_ratio))]
    
    
    train.to_csv(cwd+'\\data\\training_dataset.csv')
    test.to_csv(cwd+ '\\data\\testing_dataset.csv')
    valid.to_csv(cwd+ '\\data\\validation_dataset.csv')
    return train,test,valid

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

# Models -------------------------------------------------------------------------------------------
def LSTM_construction(layer_num=None,layer_units=None,input_shape=12,output_units=1,activation='relu'):

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

def DNN_construction(layer_num=None,layer_units=None,input_shape=12,output_units=1,activation='relu'):
    """
    input_layer : number of denses in input layer.
    hidden_layer: number of denses in hideen layers.
    hidden_num : number of hidden layers.
    activation_func: activation function.
    input_shape : window_size ( sample size)
    Return :DNN model
    """
    if layer_num == 1:
        DNN = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    input_shape, activation=activation, input_shape=[input_shape]
                ),
                tf.keras.layers.Dense(layer_units, activation=activation),
                tf.keras.layers.Dense(output_units),
            ]
        )
    if layer_num == 2:
        DNN = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    input_shape, activation, input_shape=[input_shape]
                ),
                tf.keras.layers.Dense(layer_units, activation=activation),
                tf.keras.layers.Dense(layer_units, activation=activation),
                tf.keras.layers.Dense(output_units),
            ]
        )
    if layer_num == 3:
        DNN = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    input_shape, activation=activation, input_shape=[input_shape]
                ),
                tf.keras.layers.Dense(layer_units, activation=activation),
                tf.keras.layers.Dense(layer_units, activation=activation),
                tf.keras.layers.Dense(layer_units, activation=activation),
                tf.keras.layers.Dense(output_units),
            ]
        )
    return DNN

def RNN_construction(layer_num=None,layer_units=None,input_shape=12,output_units=1,activation='relu'):
    """

    hidden_layer: number of denses in hideen layers.
    hidden_num : number of hidden layers.
    activation_func: activation function.
    input_shape : window_size ( sample size)
    Return :RNN model
    """
    if layer_num == 1:
        simple_RNN = tf.keras.models.Sequential(
            [
                tf.keras.layers.Lambda(
                    lambda x: tf.expand_dims(x, axis=-1), input_shape=[input_shape]
                ),
                tf.keras.layers.SimpleRNN(
                    layer_units, return_sequences=True, activation=activation
                ),
                tf.keras.layers.Dense(output_units),
                # tf.keras.layers.Lambda(lambda x: x * 100.0),
            ]
        )
    if layer_num == 2:
        simple_RNN = tf.keras.models.Sequential(
            [
                tf.keras.layers.Lambda(
                    lambda x: tf.expand_dims(x, axis=-1), input_shape=[input_shape]
                ),
                tf.keras.layers.SimpleRNN(
                    layer_units, return_sequences=True, activation=activation
                ),
                tf.keras.layers.SimpleRNN(layer_units, activation=activation),
                tf.keras.layers.Dense(output_units),
                # tf.keras.layers.Lambda(lambda x: x * 100.0),
            ]
        )
    if layer_num == 3:
        simple_RNN = tf.keras.models.Sequential(
            [
                tf.keras.layers.Lambda(
                    lambda x: tf.expand_dims(x, axis=-1), input_shape=[input_shape]
                ),
                tf.keras.layers.SimpleRNN(
                    layer_units, return_sequences=True, activation=activation
                ),
                tf.keras.layers.SimpleRNN(
                    layer_units, return_sequences=True, activation=activation
                ),
                tf.keras.layers.SimpleRNN(layer_units, activation=activation),
                tf.keras.layers.Dense(output_units),
                # tf.keras.layers.Lambda(lambda x: x * 100.0),
            ]
        )
    return simple_RNN

def CNN_construction(layer_num=None,layer_units=None,input_shape=12,output_units=1,activation='relu'):
    """
    input_layer : number of denses in input layer.
    hidden_layer: number of denses in hideen layers.
    hidden_num : number of hidden layers.
    activation_func: activation function.
    input_shape : window_size ( sample size)
    Return :LASTM model
    """
    if layer_num == 1:
        CNN = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    padding="causal",
                    activation=activation,
                    input_shape=[input_shape, 1],
                ),
                tf.keras.layers.LSTM(
                    layer_units, return_sequences=True, activation=activation
                ),
                tf.keras.layers.Dense(output_units),
                # tf.keras.layers.Lambda(lambda x: x * 100.0),
            ]
        )
    if layer_num == 2:
        CNN = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    padding="causal",
                    activation=activation,
                    input_shape=[input_shape, 1],
                ),
                tf.keras.layers.LSTM(
                    layer_units, return_sequences=True, activation=activation
                ),
                tf.keras.layers.LSTM(layer_units, activation=activation),
                tf.keras.layers.Dense(output_units),
                # tf.keras.layers.Lambda(lambda x: x * 100.0),
            ]
        )
    if layer_num == 3:
        CNN = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    padding="causal",
                    activation=activation,
                    input_shape=[input_shape, 1],
                ),
                tf.keras.layers.LSTM(
                    layer_units, return_sequences=True, activation=activation
                ),
                tf.keras.layers.LSTM(
                    layer_units, return_sequences=True, activation=activation
                ),
                tf.keras.layers.LSTM(layer_units, activation=activation),
                tf.keras.layers.Dense(output_units),
                # tf.keras.layers.Lambda(lambda x: x * 100.0),
            ]
        )
    return CNN

# -----------------------------------------------------------------------------------------------------

def tune_learning_rate(model=None,dataset= None,validation_data=None,fig_name=None, title = None ,patience = 5 ,loss='mse', epochs= 100,momentum=0.9,plot = False):
    # Define the learning rate schedule
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10 ** (epoch / 20))

    # Define the optimizer and compile the model
    optimizer = tf.keras.optimizers.SGD(momentum)
    model.compile(loss=loss, optimizer=optimizer)
    
    # Define early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    
    # Train the model with the early stopping callback
    history = model.fit(dataset, epochs=epochs, validation_data=validation_data, callbacks=[lr_schedule, early_stopping])
    
    # Compute the learning rates used during training
    lrs = 1e-5 * (10 ** (np.arange(epochs) / 20))
    

    lr_loss = history.history['loss']
    
    
    if plot:
      # Plot the losses vs learning rates graph and save it to file
      epochs = len(history.history['loss'])
      plt.figure(figsize=(10, 4))
      plt.grid(True)
      plt.semilogx(lrs[:epochs], history.history["loss"])
      plt.tick_params("both", length=10, width=1, which="both")
    #   plt.axis([1e-5, 1e-1, 0, 20])
      plt.title(title)
      plt.xlabel('Learning rate')
      plt.ylabel('Loss')
      plt.savefig(cwd + f'\\results\\graphs\\{fig_name}')
      plt.close()
    
    # Select optimal learning rate
    optimal_lr = lrs[np.argmin(history.history['val_loss'])]
    print('Optimal learning rate:', optimal_lr)
    return optimal_lr ,lr_loss

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

def plot_series(x,y,fig_name=None,format="-",start=0,end=None,title=None,xlabel=None,ylabel=None,legend=None,loss=None):
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
    # plt.text(0.95, 0.01, " ** loss value = : {:0.4f}".format(loss), verticalalignment='bottom', horizontalalignment='right',color='green',fontsize=12)
   
    plt.savefig(cwd + f'\\results\\graphs\\{fig_name}')
    # Draw the graph on screen
    plt.close()

def plot_training_validation_loss(model=None,history=None):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plot_series(epochs, (loss,val_loss), fig_name=f'{model}_training_loss.png', title=f"{model} training loss",xlabel='Epochs',ylabel='Loss',legend=['Training Loss','Validation Loss'],loss = val_loss)
    
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

def models_saver(model=None,name=None):
  model.save(cwd + f"\\results\\models\\{name}_model.h5")

def models_loader (model): 
  saved_model = f"{model}_model.h5"
  model = tf.keras.models.load_model(cwd + f'\\results\\models\\{saved_model}',compile=False )
  return model