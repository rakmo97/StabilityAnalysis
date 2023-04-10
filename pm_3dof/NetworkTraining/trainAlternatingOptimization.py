# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:22:37 2020

@author: Omkar
"""
# ==================================
# Import Dependencies
# ==================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
import numpy as np
import time

from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from matplotlib import pyplot as plt

# ==================================
# Define Functions
# ==================================
def MSE(y_true, y_predicted):
    # y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_predicted = tf.cast(y_predicted, dtype=tf.float64)
    # print("y_true: {}".format(y_true))
    # print("y_predicted: {}".format(y_predicted))
    # print("y_predicted - y_true: {}".format(y_predicted - y_true))
    
    return tf.reduce_mean(tf.square(y_predicted - y_true))

@tf.function
def train_step(x, y):
    g = tf.constant(9.81, dtype=tf.float64)

    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:

        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        y_pred = TF(x, training=True)
        y_pred = tf.cast(y_pred, dtype=tf.float64)



        # Compute the loss value for this minibatch.
        mse = loss_fn(y, y_pred)

        # loss_value = mse + vdot
        loss_value = mse


    print("x: {}".format(x))
    print("y: {}".format(y))
    print("y_pred: {}".format(y_pred))
    print("mse: {}".format(mse))
    print("loss_value: {}".format(loss_value))


    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, TF.trainable_weights)
    print("TF.trainable_weights: {}".format(TF.trainable_weights))
    print("grads: {}".format(grads))
    print("zip(grads, TF.trainable_weights): {}\n\n\n\n\n\n\n\n".format(zip(grads, TF.trainable_weights)))

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    opt.apply_gradients(zip(grads, TF.trainable_weights))

    # Update training metric.
    train_acc_metric.update_state(y, y_pred)
    train_loss_metric.update_state(y, y_pred)

    return loss_value


@tf.function
def test_step(x, y):
    val_y_pred = TF(x, training=False)
    # Update val metrics
    val_acc_metric.update_state(y, val_y_pred)
    val_loss_metric.update_state(y, val_y_pred)

# ==================================
# Load Data
# ==================================
# Load in training and testing data
print("Loading mat file")
# base_data_folder = '/orange/rcstudents/omkarmulekar/StabilityAnalysis/'
base_data_folder = 'E:/Research_Data/StabilityAnalysis/'
formulation = 'pm_3dof/'
matfile = loadmat(base_data_folder+formulation+'ANN2_data.mat')
saveflag = 'custom'

Xfull = matfile['Xfull_2']
tfull = matfile['tfull_2']
X_train = matfile['Xtrain2'].reshape(-1,6)
t_train = matfile['ttrain2']
X_test = matfile['Xtest2'].reshape(-1,6)
t_test = matfile['ttest2']

# Xfull = matfile['Xfull_2'].reshape(-1,7)
# tfull = matfile['tfull_2'][:,2].reshape(-1,1)
# X_train = matfile['Xtrain2'].reshape(-1,7)
# t_train = matfile['ttrain2'][:,2].reshape(-1,1)
# X_test = matfile['Xtest2'].reshape(-1,7)
# t_test = matfile['ttest2'][:,2].reshape(-1,1)


# ==================================
# Build Network
# ==================================
    
# activation = "relu"
activation = "tanh"
#
# n_neurons = 2000
# n_neurons = 250
n_neurons = 100


# # Define ANN Architecture
# TF = Sequential()
# # TF.add(layers.BatchNormalization())
# TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal',input_dim=6))
# TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# # TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# # TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# # TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# # TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# # TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# # TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# # TF.add(layers.Dense(25, activation=activation,kernel_initializer='normal'))
# # TF.add(layers.BatchNormalization())
# TF.add(layers.Dense(t_train.shape[1], activation='linear',kernel_initializer='normal'))


inputs = keras.Input(shape=(6,))
x1 = layers.Dense(n_neurons, activation=activation, kernel_initializer='normal')(inputs)
x2 = layers.Dense(n_neurons, activation=activation, kernel_initializer='normal')(x1)
outputs = layers.Dense(3,  activation='linear')(x2)


TF = keras.Model(inputs=inputs, outputs=outputs)


print("Network: {}".format(TF.summary()))

# ==================================
# Training Settings
# ==================================

# Instantiate an optimizer.
opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
# Instantiate a loss function.
# loss_fn =  keras.losses.MeanSquaredError()
loss_fn = MSE

batch_size=100
epochs = 10000
patience = 25
wait = 0
best = float('inf')


# Prepare the metrics.
train_acc_metric = keras.metrics.MeanSquaredError()
train_loss_metric = keras.metrics.MeanSquaredError()
val_acc_metric = keras.metrics.MeanSquaredError()
val_loss_metric  = keras.metrics.MeanSquaredError()


# ==================================
# Training Loop
# ==================================

history = {'loss': [], 'val_loss': []}

# Reserve 10,000 samples for validation.
x_val = X_train[-10000:]
y_val = t_train[-10000:]
x_train = X_train[:-10000]
y_train = t_train[:-10000]

print("x_val: {}".format(x_val.shape))
print("y_val: {}".format(y_val.shape))
print("x_train: {}".format(x_train.shape))
print("y_train: {}".format(y_train.shape))

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        
        # TRAIN STEP FUNCTION
        loss_value = train_step(x_batch_train, y_batch_train)
        
        # Log every 200 batches.
        if step % 1000 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4e"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * batch_size))
        
    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    train_loss = train_loss_metric.result()
    history['loss'].append(train_loss)

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()
    train_loss_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        # TEST STEP FUNCTION
        test_step(x_batch_val, y_batch_val)

    val_acc = val_acc_metric.result()
    val_loss = val_loss_metric.result()
    history['val_loss'].append(val_loss)

    val_acc_metric.reset_states()
    val_loss_metric.reset_states()
   
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Validation Loss: %.4e" % (float(val_loss),))
    print("Time taken: %.2fs" % (time.time() - start_time))

    # The early stopping strategy: stop the training if `val_loss` does not
    # decrease over a certain number of epochs.
    wait += 1
    print("============ EPOCH SUMMARY ============")
    print("val_loss = {}".format(val_loss))
    print("best = {}".format(best))
    print("wait = {}".format(wait))
    if val_loss < best:
        print("Val loss: {} < best: {}, setting wait to 0".format(val_loss, best))
        best = val_loss
        wait = 0
    if wait >= patience:
        print("Early Stopping condition met, breaking")
        break



# ==================================
# Test and Save
# ==================================


# Plotting histories
plt.figure(1)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.legend(['train', 'validation'], loc='best')
plt.title('Loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Cost (MSE)')
plt.savefig('{}nettrain_logloss.png'.format(saveflag))


# Save model
print("\nSaving ANN!")
saveout_filename = base_data_folder + formulation + "NetworkTraining/{}ANN2_703_{}_n{}.h5".format(saveflag,activation,n_neurons)
print('Filename: ' + saveout_filename)
TF.save(saveout_filename)


#

# Plotting
print('Plotting')
# idxs = range(000,X_test.shape[0])
idxs = range(000,300)
# yvis = TF.predict(X_test[idxs].reshape(-1,7));
yvis = TF.predict(X_test[idxs].reshape(-1,6));



plt.figure(3)
plt.subplot(311)
plt.plot(t_test[idxs,0])
plt.plot(yvis[:,0],'--')
plt.xlabel('Index (-)')
plt.ylabel('Tx (N)')
plt.legend(['ocl','ann'])
plt.subplot(312)
plt.plot(t_test[idxs,1])
plt.plot(yvis[:,1],'--')
plt.xlabel('Index (-)')
plt.ylabel('Ty (N)')
plt.subplot(313)
plt.plot(t_test[idxs,2])
plt.plot(yvis[:,2],'--')
plt.xlabel('Index (-)')
plt.ylabel('Tz (N)')
plt.suptitle('OL Test: {}'.format(saveflag))
plt.tight_layout()
plt.savefig('{}nettrain_OLtest.png'.format(saveflag))


# plt.figure(3)

# plt.plot(t_test[idxs])
# plt.plot(yvis)
# plt.xlabel('Index (-)')
# plt.ylabel('Tx (N)')
# plt.legend(['ocl','ann'])

