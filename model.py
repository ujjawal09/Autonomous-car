import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
import os

np.random.seed(0)

data_dir = 'E:/project/Udacity/simulator-windows-64/IMG'
itest_size = 0.2
ikeep_prob = 0.5
ilr=0.0001
ibatch_size = 40
isamples_per_epoch = 20000
inb_epochs = 3

def load_data(data_dir, itest_size):
   data_df = pd.read_csv("driving_log.csv", names=["center","left","right","steering_angle","throttle","reverse","speed"])
   X = data_df[['center', 'left', 'right']].values
   y = data_df["steering_angle"].values
   X_train,X_valid, y_train, y_valid = train_test_split(X, y, test_size=itest_size, random_state=0)
   return X_train, X_valid, y_train, y_valid

def build_model(ikeep_prob):
    model = Sequential() 
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(ikeep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model

def train_model(model,ilr, ibatch_size, isamples_per_epoch, inb_epochs, data_dir, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint('model-{epoch:03d}-{val_loss:.02f}.h5',# if filepath is weights.{epoch:02d}-{val_loss:.2f}.hdf5, then the model checkpoints will be saved with the epoch number and the validation loss in the filename.
                                 monitor='val_loss',
                                 verbose = 0,#0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
                                 save_best_only = True,#save_best_only: if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.

                                                       
                                     mode = 'auto') #mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity.
                                                         #For val_acc, this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity.
    tensorboard = TensorBoard(log_dir='./logs', write_graph=True)
    
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=ilr))

    model.fit_generator(batch_generator(data_dir, X_train, y_train, ibatch_size, True),
                        isamples_per_epoch, inb_epochs,
                        validation_data=batch_generator(data_dir, X_valid, y_valid, ibatch_size, False),
                        callbacks=[checkpoint,tensorboard],verbose=1, max_queue_size=1,validation_steps=len(X_valid))#A callback is a set of functions to be applied at given stages of the training procedure.
                                                                                                        #You can use callbacks to get a view on internal states and statistics of the model during training.
                                                                                                        #The logs dictionary that callback methods take as argument will contain keys for quantities relevant to the current batch or epoch.

    score=model.evaluate(x_valid,y_valid)
    print(score)                     

def main():
    data = load_data(data_dir, itest_size)
    model = build_model(ikeep_prob)
    train_model(model,ilr, ibatch_size, isamples_per_epoch, inb_epochs, data_dir ,*data)

if __name__ == '__main__':
    main()
                        
                                        
