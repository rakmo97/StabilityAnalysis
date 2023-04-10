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
@tf.function
def MSE(y_true, y_predicted):
    y_predicted = tf.cast(y_predicted, dtype=tf.float64)   
    return tf.reduce_mean(tf.square(y_predicted - y_true))

@tf.function
def MeanVdot(x, y_predicted):
    g = tf.constant(9.81, dtype=tf.float64)

    x = tf.cast(x, dtype=tf.float64)   
    y_predicted = tf.cast(y_predicted, dtype=tf.float64)   

    return tf.reduce_mean(x[:,0]*x[:,3] + x[:,1]*x[:,4] + x[:,2]*x[:,5] + x[:,3]*y_predicted[:,0] + x[:,4]*y_predicted[:,1] + x[:,5]*(y_predicted[:,2] - g))

@tf.function
def MeanLagrangian(x, y_true, y_predicted):

    L = MSE(y_true, y_predicted) + multiplier*MeanVdot(x, y_predicted)

    return L




@tf.function
def min_step(x, y):

    # Gradient tape for autodiff
    with tf.GradientTape() as tape:

        # Forward Pass
        y_pred = TF(x, training=True)
        y_pred = tf.cast(y_pred, dtype=tf.float64)


        # Compute the lagrangian value for this minibatch.
        L_b = MeanLagrangian(x, y, y_pred)



    # Calculate gradients wrt weights
    grads = tape.gradient(L_b, TF.trainable_weights)

    #  Apply gradients
    opt_min.apply_gradients(zip(grads, TF.trainable_weights))

    # Update training metric.
    # train_loss_metric.update_state(x, y, y_pred)
    train_acc_metric.update_state(y, y_pred)

    return L_b


@tf.function
def test_step(x, y):
    val_y_pred = TF(x, training=False)
    # Update val metrics
    val_loss = MeanLagrangian(x, y, val_y_pred)
    val_acc_metric.update_state(y, val_y_pred)

    return val_loss

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
opt_min = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
opt_max = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)


# Batch/early stopping parameters
batch_size=100
epochs = 10000
patience = 25
wait = 0
best = float('inf')


# Prepare the metrics.
train_loss_metric = MeanLagrangian
val_loss_metric  = MeanLagrangian


train_acc_metric = keras.metrics.MeanSquaredError()
val_acc_metric  = keras.metrics.MeanSquaredError()


# ==================================
# Training Loop
# ==================================

history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'multiplier': []}

# Reserve 10,000 samples for validation.
x_val = X_train[-10000:]
y_val = t_train[-10000:]
# x_train = X_train[:-10000]
# y_train = t_train[:-10000]

x_train = X_train[:-4500000]
y_train = t_train[:-4500000]


# Prepare the training dataset.
train_dataset_full = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset_full_batch = train_dataset_full.batch(X_train.shape[0])
train_dataset = train_dataset_full.shuffle(buffer_size=1024).batch(batch_size)


# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)

# Lagrange multiplier
multiplier = [tf.Variable(0.1, dtype=tf.float64)]

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()


    # MAX STEP
    for x_full_train, y_full_train in train_dataset_full_batch:

        # Max step funciton
        # multiplier = max_step(x_full_train, y_full_train)
        # Forward Pass
        y_pred = TF(x_full_train, training=True)
        y_pred = tf.cast(y_pred, dtype=tf.float64)
    
        # Calculate gradients wrt multiplier (analytical form, not autodiff)
        grads = [MeanVdot(x_full_train, y_pred)]
    
    
        #  Apply gradients
        opt_max.apply_gradients(zip(grads, multiplier))

        
    
    multiplierTensor = tf.constant(multiplier[0])
    history['multiplier'].append(multiplierTensor)

    # MIN STEP
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        
        # Min step function
        loss_value = min_step(x_batch_train, y_batch_train)
        
        # Log every 200 batches.
        # if step % 10000 == 0:
        #     print(
        #         "Training loss (for one batch) at step %d: %.4e"
        #         % (step, float(loss_value))
        #     )
        #     print("Seen so far: %s samples" % ((step + 1) * batch_size))
        
    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    history['loss'].append(loss_value)
    history['acc'].append(train_acc)

    # Reset training metrics at the end of each epoch
    # train_loss_metric.reset_states()
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        # TEST STEP FUNCTION
        val_loss = test_step(x_batch_val, y_batch_val)

    val_acc = val_acc_metric.result()
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    val_acc_metric.reset_states()
   
    print("Multiplier Value: %.4f" % (float(multiplierTensor),))
    print("Training Loss epoch (Lagrangian): %.4f" % (float(loss_value),))
    print("Validation Loss epoch (Lagrangian): %.4f" % (float(val_loss),))
    print("Training Metric epoch (MSE): %.4f" % (float(train_acc),))
    print("Validation Metric (MSE))): %.4e" % (float(val_acc),))
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
plt.plot(np.array(history['loss']).reshape(-1))
plt.plot(np.array(history['val_loss']).reshape(-1))
plt.legend(['train', 'validation'], loc='best')
plt.title('Loss (Lagrangian)')
# plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Cost (MSE)')
plt.savefig('{}nettrain_logloss.png'.format(saveflag))

plt.figure(2)
plt.plot(np.array(history['acc']))
plt.plot(np.array(history['val_acc']))
plt.legend(['train', 'validation'], loc='best')
plt.title('Accuracy (MSE)')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Cost (MSE)')
plt.savefig('{}nettrain_logacc.png'.format(saveflag))

plt.figure(3)
plt.plot(np.array(history['multiplier']))
plt.legend(['Lagrange Multiplier'], loc='best')
plt.title('Accuracy (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Cost (MSE)')
plt.savefig('{}nettrain_lagrangemultiplier.png'.format(saveflag))


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

