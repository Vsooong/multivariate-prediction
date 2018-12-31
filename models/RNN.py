from keras.models import Sequential
from keras.layers import *
import torch.nn as nn
class Model(nn.Module):

    def __init__(self, args, data):
        super(Model, self).__init__()
        self.windows=args.window
        self.variables = data.m
        self.model=self.build_keras_model()

    def build_keras_model(self):

        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(self.windows, self.variables)))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.variables))
        model.compile(loss='mse', optimizer='adam')
        return model
    def forward(self, x):
        return self.model(x)
