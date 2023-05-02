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
def MeanVdot_negative(x, y_predicted):
    g = tf.constant(9.81, dtype=tf.float64)

    x = tf.cast(x, dtype=tf.float64)   
    y_predicted = tf.cast(y_predicted, dtype=tf.float64)   

    return -tf.reduce_mean(x[:,0]*x[:,3] + x[:,1]*x[:,4] + x[:,2]*x[:,5] + x[:,3]*y_predicted[:,0] + x[:,4]*y_predicted[:,1] + x[:,5]*(y_predicted[:,2] - g))

@tf.function
def MaxVdot_negative(x, y_predicted):
    g = tf.constant(9.81, dtype=tf.float64)

    x = tf.cast(x, dtype=tf.float64)   
    y_predicted = tf.cast(y_predicted, dtype=tf.float64)   

    return -tf.reduce_max(x[:,0]*x[:,3] + x[:,1]*x[:,4] + x[:,2]*x[:,5] + x[:,3]*y_predicted[:,0] + x[:,4]*y_predicted[:,1] + x[:,5]*(y_predicted[:,2] - g))


@tf.function
def MeanAugmentedLagrangian(x, y_true, y_predicted, mu):

    # L = MSE(y_true, y_predicted) - multiplier*(MeanVdot_negative(x, y_predicted) - slack) + (1/(2*mu))*(MeanVdot_negative(x, y_predicted) - slack)**2
    L = MSE(y_true, y_predicted) - multiplier*(MaxVdot_negative(x, y_predicted) - slack) + (1/(2*mu))*(MaxVdot_negative(x, y_predicted) - slack)**2

    return L




@tf.function
def min_step(x, y):
    
    # Gradient tape for autodiff
    with tf.GradientTape() as tape:
        tape.watch(slack)
        # Forward Pass
        y_pred = TF(x, training=True)
        y_pred = tf.cast(y_pred, dtype=tf.float64)


        # Compute the lagrangian value for this minibatch.
        L_b = MeanAugmentedLagrangian(x, y, y_pred, mu)


    # Calculate gradients wrt weights
    grads = tape.gradient(L_b, [TF.trainable_weights,slack])


    #  Apply gradients
    # opt_min.apply_gradients(zip(grads, [TF.trainable_weights,slack]))
    opt_min.apply_gradients(zip(grads[0], TF.trainable_weights))
    opt_min.apply_gradients(zip(grads[1], slack))

    # Update training metric.
    # train_loss_metric.update_state(x, y, y_pred)
    train_acc_metric.update_state(y, y_pred)

    return L_b


@tf.function
def test_step(x, y):
    val_y_pred = TF(x, training=False)
    # Update val metrics
    val_loss = MeanAugmentedLagrangian(x, y, val_y_pred,mu)
    val_acc_metric.update_state(y, val_y_pred)
    val_max_vdot = -MaxVdot_negative(x_batch_val, val_y_pred)


    return val_loss, val_max_vdot

# ==================================
# Load Data
# ==================================
# Load in training and testing data
print("Loading mat file")
base_data_folder = '/orange/rcstudents/omkarmulekar/StabilityAnalysis/'
# base_data_folder = 'E:/Research_Data/StabilityAnalysis/'
formulation = 'pm_3dof/'
matfile = loadmat(base_data_folder+formulation+'ANN2_data.mat')
saveflag = 'lancelot_max_100episodes'

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
# Build Network
# ==================================
    
# activation = "relu"
activation = "tanh"
#
# n_neurons = 2000
# n_neurons = 250
n_neurons = 100


# # Define ANN Architecture
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
batch_size=1000
epochs_min = 10000
epochs_max = 1
episodes = 100
patience = 25
wait = 0
best = float('inf')


# Prepare the metrics.
train_loss_metric = MeanAugmentedLagrangian
val_loss_metric  = MeanAugmentedLagrangian


train_acc_metric = keras.metrics.MeanSquaredError()
val_acc_metric  = keras.metrics.MeanSquaredError()


# ==================================
# Training Loop
# ==================================

history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'multiplier': [], 'slack': [], 'mu': [], 'alpha': [], 'omega': [], 'eta': [], 'constraint': []}

# Reserve 10,000 samples for validation.
x_val = X_train[-10000:]
y_val = t_train[-10000:]
# x_train = X_train[:-10000]
# y_train = t_train[:-10000]
x_train = X_train[:1000000]
y_train = t_train[:1000000]

# x_train = X_train[:-4500000]
# y_train = t_train[:-4500000]
print("Utelized training size: {}".format(x_train.shape[0]))


# Prepare the training dataset.
train_dataset_full = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset_full_batch = train_dataset_full.batch(X_train.shape[0])
train_dataset = train_dataset_full.shuffle(buffer_size=1024).batch(batch_size)


# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)

# Lagrange multiplier
multiplier = [tf.Variable(10.0, dtype=tf.float64)]
slack = [tf.Variable(0.0001, dtype=tf.float64)]
etabar = 0.1
omegabar = 0.1
mubar = 1.0
tau = 0.1
gammabar = 0.1
alphaomega = 0.1
betaomega = 0.1
alphaeta = 0.05
betaeta = 0.05
alphastar = 0.1
betastar = 0.1
etastar = 0.001

val_max_vdot = 0

mu = mubar
alpha = min(mu,gammabar)
omega = omegabar*alpha**alphaomega
eta = etabar*alpha**alphaeta

slackTensor = tf.constant(slack[0])
history['slack'].append(slackTensor)
history['mu'].append(mu)
history['alpha'].append(alpha)
history['eta'].append(eta)
history['omega'].append(omega) 
multiplierTensor = tf.constant(multiplier[0])
history['multiplier'].append(multiplierTensor)

for episode in range(episodes):
    wait = 0

    print("==============================")
    print("==============================")
    print("\nStart of episode {} of {}".format(episode,episodes))

    print('Minimization Step')
    for epoch in range(epochs_min):
        print("Start of epoch {} (episode {}".format(epoch,episode))
        start_time = time.time()

        # MIN STEP
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            
            # Min step function
            loss_value = min_step(x_batch_train, y_batch_train)



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
            val_loss, val_max_vdot = test_step(x_batch_val, y_batch_val)

        val_acc = val_acc_metric.result()
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        val_acc_metric.reset_states()

        # The early stopping strategy: stop the training if `val_loss` does not
        # decrease over a certain number of epochs.
        wait += 1
        print("--- Epoch Summary ---")
        print("Training Loss epoch (Lagrangian): %.4f" % (float(loss_value),))
        print("Validation Loss epoch (Lagrangian): %.4f" % (float(val_loss),))
        print("Training Metric epoch (MSE): %.4f" % (float(train_acc),))
        print("Validation Metric (MSE))): %.4e" % (float(val_acc),))
        print("Max Vdot: {}".format(val_max_vdot))
        print("slack: {}".format(slack))
        print("Time taken: %.2fs" % (time.time() - start_time))
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

    for x_max_step, y_max_step in train_dataset_full_batch:
        
        y_pred = TF(x_max_step, training=True)
        # c = MeanVdot_negative(x_max_step, y_pred) - slack
        c = MaxVdot_negative(x_max_step, y_pred) - slack
        

    if tf.norm(c) < eta:
        if tf.norm(c) < etastar: # Test for convergence
            break
        
        #Update multipliers, tighten tolerances
        # multiplier[0].assign_sub((c/mu).numpy().tolist()[0])
        multiplier[0].assign(tf.math.maximum(0,multiplier[0] - (c/mu))[0])        
        alpha = mu
        eta = eta*(alpha**betaeta)
        omega = omega*(alpha**betaomega)
        
    else:
        # decrease penalty parameter, tighten tolerances
        mu = tau*mu
        alpha - mu*gammabar
        eta = etabar*(alpha**betaeta)
        omega = omegabar*(alpha**betaomega)


    print("\n\n\n------ Episode Summary ------")
    print('constraint: {}'.format(c))
    print('multiplier: {}'.format(multiplier))
    print('slack: {}'.format(slack))
    print('mu: {}'.format(mu))
    print('alpha: {}'.format(alpha))
    print('eta: {}'.format(eta))
    print('omega: {}'.format(omega))

    slackTensor = tf.constant(slack[0])
    history['slack'].append(slackTensor)
    history['mu'].append(mu)
    history['alpha'].append(alpha)
    history['eta'].append(eta)
    history['omega'].append(omega)   
    multiplierTensor = tf.constant(multiplier[0])
    history['multiplier'].append(multiplierTensor)
    history['constraint'].append(c)


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

plt.figure(4)
plt.plot(np.array(history['constraint']))
plt.legend(['Constraint value'], loc='best')
plt.title(' ')
plt.xlabel('Epochs')
plt.ylabel('Constraint Value')
plt.savefig('{}nettrain_constraint_value.png'.format(saveflag))


plt.figure(5)
plt.plot(np.array(history['slack']))
plt.legend(['Slack value'], loc='best')
plt.title(' ')
plt.xlabel('Episodes')
plt.ylabel('Slack Value')
plt.savefig('{}nettrain_slack_value.png'.format(saveflag))



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



plt.figure(6)
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

