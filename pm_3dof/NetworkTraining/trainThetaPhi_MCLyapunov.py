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
from tensorflow.keras import layers, models
from tensorflow.keras.losses import Loss
import numpy as np
import LyapunovNetwork as LN
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

# @tf.function
def MaxVdot(x, u):
    g = tf.constant(9.81, dtype=tf.float64)

    x = tf.cast(x, dtype=tf.float64)   
    u = tf.cast(u, dtype=tf.float64)   
    u = tf.clip_by_value(u, (-20,-20,0),(20,20,20))

    xdot = np.array([x[:,3], x[:,4], x[:,5], u[:,0], u[:,1], u[:,2]-g]).T
    xdot = tf.constant(xdot, dtype=tf.float64)
    
    with tf.GradientTape() as tape_phi:
        tape_phi.watch(x)
        V_pred = V_phi(x)

    dVdx = tape_phi.gradient(V_pred, x)
    dVdx = tf.reshape(dVdx, (-1,6))

    Vdot = tf.einsum('ij, ij->i', dVdx, xdot)
    return tf.reduce_max(Vdot)


# @tf.function
def Lagrangian(x, y_true, y_predicted, x_MC, y_MC):

    L = MSE(y_true, y_predicted) + multiplier*MaxVdot(x_MC, y_MC)

    return L




# @tf.function
def min_step(x_batch, y_true, x_MC):
    
    # Gradient tape for autodiff
    with tf.GradientTape() as tape:
        # Forward Pass
        y_pred_batch = TF(x_batch, training=True)
        y_pred_batch = tf.cast(y_pred_batch, dtype=tf.float64)

        y_pred_MC = TF(x_MC, training=True)
        y_pred_MC = tf.cast(y_pred_MC, dtype=tf.float64)

        # Compute the lagrangian value for this minibatch.
        train_loss = Lagrangian(x_batch, y_true, y_pred_batch, x_MC, y_pred_MC)


    # Calculate gradients wrt weights
    grads = tape.gradient(train_loss, [TF.trainable_weights,V_phi.trainable_weights])


    #  Apply gradients
    opt_theta.apply_gradients(zip(grads[0], TF.trainable_weights))
    opt_phi.apply_gradients(zip(grads[1], V_phi.trainable_weights))



    # Update training metric.
    train_MSE = MSE(y_true, y_pred_batch)
    train_max_vdot = MaxVdot(x_MC, y_pred_MC)


    return train_loss, train_MSE, train_max_vdot


# @tf.function
def test_step(x, y, x_MC):
    val_y_pred = TF(x, training=False)
    val_y_pred_MC = TF(x_MC, training=False)

    # Update val metrics
    val_loss = Lagrangian(x, y, val_y_pred, x_MC, val_y_pred_MC)
    val_MSE = MSE(y, val_y_pred)

    val_max_vdot = MaxVdot(x_MC, val_y_pred_MC)


    return val_loss, val_MSE, val_max_vdot

# ==================================
# Load Data
# ==================================
# Load in training and testing data
print("Loading mat file")
base_data_folder = '/orange/rcstudents/omkarmulekar/StabilityAnalysis/'
# base_data_folder = 'E:/Research_Data/StabilityAnalysis/'
formulation = 'pm_3dof/'
matfile = loadmat(base_data_folder+formulation+'ANN2_data.mat')
saveflag = 'trainThetaPhi_MCLyapunov_10break_normal_'
print('\nRUNNING PROGRAM USING SAVE FLAG     {}\n'.format(saveflag))

Xfull = matfile['Xfull_2']
tfull = matfile['tfull_2']
X_train = matfile['Xtrain2'].reshape(-1,6)
t_train = matfile['ttrain2']
X_test = matfile['Xtest2'].reshape(-1,6)
t_test = matfile['ttest2']

print("Full training size: {}".format(X_train.shape[0]))

# Xfull = matfile['Xfull_2'].reshape(-1,7)
# tfull = matfile['tfull_2'][:,2].reshape(-1,1)
# X_train = matfile['Xtrain2'].reshape(-1,7)
# t_train = matfile['ttrain2'][:,2].reshape(-1,1)
# X_test = matfile['Xtest2'].reshape(-1,7)
# t_test = matfile['ttest2'][:,2].reshape(-1,1)


# ==================================
# Build Networks
# ==================================
    
# activation = "relu"
activation = "tanh"
#
# n_neurons = 2000
# n_neurons = 250
n_neurons = 100


# # Define ANN Architecture
# inputs = keras.Input(shape=(6,))
# x1 = layers.Dense(n_neurons, activation=activation, kernel_initializer='normal')(inputs)
# x2 = layers.Dense(n_neurons, activation=activation, kernel_initializer='normal')(x1)
# x3 = layers.Dense(n_neurons, activation=activation, kernel_initializer='normal')(x2)
# outputs = layers.Dense(3,  activation='linear')(x3)
# TF = keras.Model(inputs=inputs, outputs=outputs)
policy_filename = base_data_folder+formulation+'NetworkTraining/customANN2_703_tanh_n100.h5'
TF = models.load_model(policy_filename)
print("Policy Network: {}".format(TF.summary()))

# Load in trained Lyapunov Network
lyapunov_filename = base_data_folder + formulation + "NetworkTraining/MinimizedLyapunovNetwork_MSE_LyBasic_Max.h5"
V_phi = models.load_model(lyapunov_filename,  custom_objects={'LyapunovDense': LN.LyapunovDense})
test_V_phi = V_phi.predict(np.array([[1,1,1,1,1,1]]))
print('test_V_phi: {}'.format(test_V_phi))


# ==================================
# Training Settings
# ==================================

# Instantiate an optimizer.
opt_theta = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
opt_phi   = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)


# Batch/early stopping parameters
batch_size=10000
epochs_min = 10000
episodes = 100
MC_num_train = 100000
MC_num_val = 1000000

episode_break_condition_counter = 0
episode_break_condition = 10
patience = 8
wait = 0
best = float('inf')
val_MSE_last = float('inf')

# System Dynamics settings
nState    =   6
nCtrl     =   3


# Bounds for random state generation
lower = [-5, -5,   20, -5, -5, -5]
upper = [ 5,  5,  20,  5,  5,  5]
# lower = [-1, -1,   0, -1, -1, -1]
# upper = [ 1,  1,  1,  1,  1,  1]

# ==================================
# Training Loop
# ==================================

history = {'train_loss': [], 'train_MSE': [], 'train_max_vdot': [], 
           'val_loss': [], 'val_MSE': [], 'val_max_vdot': [], 
           'multiplier': []}

# Reserve 10,000 samples for validation.
x_val = X_train[-10000:]
y_val = t_train[-10000:]
# x_train = X_train[:-10000]
# y_train = t_train[:-10000]
x_train = X_train[:1000000]
y_train = t_train[:1000000]

X_MC_np = np.empty((0,6)) # initial MC states


print("Utilized training size: {}".format(x_train.shape[0]))


# Prepare the training dataset.
train_dataset_full = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset_full_batch = train_dataset_full.batch(X_train.shape[0])
train_dataset = train_dataset_full.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(x_val.shape[0])

multiplier = 0
history['multiplier'].append(multiplier)

for episode in range(episodes):

    print('\n\n-------------------------------------')
    print('-------------------------------------')
    print("Episode {} of {}".format(episode+1, episodes))

    print("-------------------------------------")
    print("Minimization Loop (Oracle)")
    
    # Reset early stopping iterators
    wait = 0
    best = float('inf')

    # Generate MC states for constraint evaluation
    # X_MC_np = np.vstack((X_MC_np, np.random.uniform(low=lower,high=upper,size=(MC_num_train,nState))))
    X_MC_np = np.vstack((X_MC_np, np.random.randn(MC_num_train,nState)*np.array([3,3,10,3,3,3])))
    # X_MC_np = np.vstack((X_MC_np, np.random.randn(MC_num_train,nState)*np.array([0.5,0.5,0.5,0.5,0.5,0.5])))
    X_MC_tensor = tf.constant(X_MC_np)
    print('Size of MC training set: {}'.format(X_MC_np.shape[0]))

    # X_MC_np_val = np.random.uniform(low=lower,high=upper,size=(MC_num_val,nState))
    X_MC_np_val = np.random.randn(MC_num_val,nState)*np.array([3,3,10,3,3,3])
    X_MC_tensor_val = tf.constant(X_MC_np_val)


    for epoch in range(epochs_min):
        print("Start of epoch {} of {}".format(epoch+1,epochs_min))
        start_time = time.time()

        # MIN STEP
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            
            # Min step function
            train_loss, train_MSE, train_max_vdot = min_step(x_batch_train, y_batch_train, X_MC_tensor)

        # Display metrics at the end of each epoch.
        history['train_loss'].append(train_loss)
        history['train_MSE'].append(train_MSE)
        history['train_max_vdot'].append(train_max_vdot)


        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            # TEST STEP FUNCTION
            val_loss, val_MSE, val_max_vdot = test_step(x_batch_val, y_batch_val, X_MC_tensor_val)


        history['val_loss'].append(val_loss)
        history['val_MSE'].append(val_MSE)
        history['val_max_vdot'].append(val_max_vdot)

        # The early stopping strategy: stop the training if `val_loss` does not
        # decrease over a certain number of epochs.
        wait += 1
        print("--- Epoch {} Summary (Epidsode {}) ---".format(epoch, episode))
        print(" train_MSE:               {}".format(train_MSE))
        print(" train_max_vdot:        {}".format(train_max_vdot))
        print(" train_loss (lagrangian): {}\n".format(train_loss))
        print(" val_MSE:               {}".format(val_MSE))
        print(" val_max_vdot:        {}".format(val_max_vdot))
        print(" val_loss (lagrangian): {}\n".format(val_loss))
        print(" multiplier: {}".format(multiplier))

        print("Time taken: %.2fs" % (time.time() - start_time))
        print("best = {}".format(best))
        print("wait = {}".format(wait))
        if val_loss < best:
            print("Val loss: {} < best: {}, setting wait to 0".format(val_loss, best))
            best = val_loss
            wait = 0
        if wait >= patience:
            print("Early Stopping condition met, breaking")
            break


    # Calculate gradient wrt lambda, update M matrix
    multiplier += val_max_vdot
    multiplier = max(0,multiplier) # Enforces multiplier >= 0
    history['multiplier'].append(multiplier)

    


    print('-------------------------------------')
    print('-------------------------------------')
    print('      EPISODE {} SUMMARY:    '.format(episode+1))
    print('dLdlam: {}'.format(val_max_vdot))
    print('multiplier: {}'.format(multiplier))
    
    if ( (val_max_vdot<0) and (val_MSE<0.5) ):
        print("Incrementing episode_break_condition_counter")
        episode_break_condition_counter += 1
    else:
        print("Setting episode_break_condition_counter to zero")
        episode_break_condition_counter = 0

    if episode_break_condition_counter >= episode_break_condition:
        print('Training conditions met, breaking.')
        break


# ==================================
# Test and Save
# ==================================




# Plotting histories
plt.figure(1)
plt.plot(np.array(history['train_loss']).reshape(-1))
plt.plot(np.array(history['val_loss']).reshape(-1))
plt.legend(['train', 'validation'], loc='best')
plt.title('Loss (Lagrangian)')
# plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Cost (MSE)')
plt.savefig('{}nettrain.png'.format(saveflag))

plt.figure(2)
plt.plot(np.array(history['train_MSE']))
plt.plot(np.array(history['val_MSE']))
plt.legend(['train', 'validation'], loc='best')
plt.title('MSE')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('MSE [-]')
plt.savefig('{}nettrain_logmse.png'.format(saveflag))

plt.figure(3)
plt.plot(np.array(history['multiplier']))
plt.title('Multiplier (-)')
plt.xlabel('Epochs')
plt.ylabel('Multiplier (-)')
plt.savefig('{}nettrain_lagrangemultiplier.png'.format(saveflag))

plt.figure(6)
plt.plot(np.array(history['train_max_vdot']))
plt.plot(np.array(history['val_max_vdot']))
plt.legend(['train', 'validation'], loc='best')
plt.title('max_vdot')
plt.xlabel('Epochs')
plt.ylabel('max_vdot [-]')
plt.savefig('{}nettrain_max_vdot.png'.format(saveflag))

# Save model
print("\nSaving Policy!")
saveout_filename = base_data_folder + formulation + "NetworkTraining/{}ANN2_703_{}_n{}_batch{}.h5".format(saveflag,activation,n_neurons, batch_size)
print('Filename: ' + saveout_filename)
TF.save(saveout_filename)

print("\nSaving Lyapunov!")
saveout_filename = base_data_folder + formulation + "NetworkTraining/{}_batch{}_Lyappunov.h5".format(saveflag, batch_size)
print('Filename: ' + saveout_filename)
V_phi.save(saveout_filename)


# Plotting
print('Plotting')
# idxs = range(000,X_test.shape[0])
idxs = range(000,300)
yvis = TF.predict(X_test[idxs].reshape(-1,6))



plt.figure(7)
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


