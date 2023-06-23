from tensorflow import keras
import tensorflow as tf

import numpy as np
import LyapunovNetwork as LN
from tensorflow.keras import Model, Input, models, layers
from tensorflow.keras.layers import Dot
from scipy.io import loadmat
from scipy.io import savemat
from matplotlib import pyplot as plt
import LanderDynamics as LD
import time


nState    =   6
nCtrl     =   3
g = 9.81


print("Loading mat file")
# base_data_folder = 'E:/Research_Data/StabilityAnalysis/'
base_data_folder = '/orange/rcstudents/omkarmulekar/StabilityAnalysis/'
formulation = 'pm_3dof/'
matfile = loadmat(base_data_folder+formulation+'ANN2_data.mat')
saveflag = 'minP_'

print('\nRUNNING PROGRAM USING SAVE FLAG     {}\n'.format(saveflag))


X_train = matfile['Xtrain2'].reshape(-1,6)
t_train = matfile['ttrain2']
# t_train = np.einsum('ij, ij->i', X_train, X_train)



# Batch/early stopping parameters
epochs = 5000
batch_size=1000


opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)



# ==================================
# Training Loop
# ==================================

history = {'train_loss': [], 'val_loss': [], 'train_Vdot_history': [], 'val_Vdot_history': []}

# Reserve 10,000 samples for validation.
x_val = tf.constant(X_train[-100000:], dtype=tf.float32)
y_val = tf.constant(t_train[-100000:], dtype=tf.float32)
# x_train = X_train[:-10000]
# y_train = t_train[:-10000]

x_train = tf.constant(X_train[:2000000], dtype=tf.float32)
y_train = tf.constant(t_train[:2000000], dtype=tf.float32)

num_train = x_train.shape[0]

print("x_val: {}".format(x_val.shape))
print("x_train: {}".format(x_train.shape))
print('Training on {} datapoints'.format(x_train.shape[0]))

# Prepare the training dataset.
train_dataset_full = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset_full_batch = train_dataset_full.batch(X_train.shape[0])
train_dataset = train_dataset_full.shuffle(buffer_size=1024).batch(batch_size)


Atensor = tf.eye(nState)
A = tf.Variable(Atensor)

for epoch in range(epochs):

    print('epoch {} of {}'.format(epoch,epochs))

    # for s in range(num_train):

    #     xdot = np.array([x_train[s,3], x_train[s,4], x_train[s,5], y_train[s,0], y_train[s,1], y_train[s,2]-g]).reshape(-1,1)
    #     xdot = tf.constant(xdot, dtype=tf.float32)

    #     with tf.GradientTape() as tape:
    #         tape.watch(A)
    #         Vdot = tf.linalg.matmul(tf.reshape(x_train[s,:],(1,-1)), tf.linalg.matmul(tf.transpose(A), tf.linalg.matmul(A, xdot)))
            
        
    #     dVdot_dA = tape.gradient(Vdot, A)
        
    #     opt.apply_gradients(zip([dVdot_dA], [A]))

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        xdot = np.array([x_batch_train[:,3], x_batch_train[:,4], x_batch_train[:,5], y_batch_train[:,0], y_batch_train[:,1], y_batch_train[:,2]-g])
        xdot = tf.constant(xdot, dtype=tf.float32)
        
        with tf.GradientTape() as tape:

            # Vdot = tf.tf.einsum('ij, ij->i', tf.linalg.matmul(x_batch_train, A), tf.linalg.matmul(tf.transpose(xdot),tf.transpose(A)))
            # Vdot = tf.reduce_mean(tf.einsum('ij, ij->i', tf.linalg.matmul(x_batch_train, A), tf.linalg.matmul(tf.transpose(xdot),tf.transpose(A))))
            Vdot = tf.reduce_max(tf.einsum('ij, ij->i', tf.linalg.matmul(x_batch_train, A), tf.linalg.matmul(tf.transpose(xdot),tf.transpose(A))))
        
        dVdot_dA = tape.gradient(Vdot, A)
        opt.apply_gradients(zip([dVdot_dA], [A]))

    # Validation step?
    xdot = np.array([x_val[:,3], x_val[:,4], x_val[:,5], y_val[:,0], y_val[:,1], y_val[:,2]-g])
    xdot = tf.constant(xdot, dtype=tf.float32)
    Vdot = tf.einsum('ij, ij->i', tf.linalg.matmul(x_val, A), tf.linalg.matmul(tf.transpose(xdot),tf.transpose(A)))
    Vdot = Vdot.numpy()

    print('\t Val Vdot.max(): {}'.format(Vdot.max()))
    positive_idx = np.argwhere(Vdot > 0)

    if 100*positive_idx.shape[0]/x_val.shape[0] < 0.01:
        print('abcd')
        break

    if Vdot.max() < 0:
        break

    print('\t% Positive Vdots: {} %'.format(100*positive_idx.shape[0]/x_val.shape[0]))

print(A)
Anumpy = A.numpy()


mdic = {"A": Anumpy}
savemat("matlab_matrix.mat", mdic)