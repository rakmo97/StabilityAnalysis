
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

class LyapunovNetwork(Model):
  def __init__(self):
    super(LyapunovNetwork, self).__init__()
    self.d1 = Dense(128, activation='tanh')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.flatten(x)
    x = self.d1(x)
    return tensordot(self.d2(x), self.d2(x))

# Create an instance of the model
model = LyapunovNetwork()


inputs = keras.Input(shape=(6,))
x1 = layers.Dense(n_neurons, activation=activation, kernel_initializer='normal')(inputs)
x2 = layers.Dense(n_neurons, activation=activation, kernel_initializer='normal')(x1)
outputs = layers.Dense(3,  activation='linear')(x2)
TF = keras.Model(inputs=inputs, outputs=outputs)


print('model: {}'.format(model))

