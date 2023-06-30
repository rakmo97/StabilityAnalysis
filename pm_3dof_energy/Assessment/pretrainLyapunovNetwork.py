# ==================================
# Import Dependencies
# ==================================
from tensorflow import keras
import tensorflow as tf

import numpy as np
import LyapunovNetwork as LN
from tensorflow.keras import Model, Input, models, layers
from tensorflow.keras.layers import Dot
from scipy.io import loadmat
from matplotlib import pyplot as plt
import LanderDynamics as LD
import time
# from tensorflow_addons.layers import SpectralNormalization

# ==================================
# Dynamics Settings
# ==================================
nState    =   6
nCtrl     =   3
g = 9.81


# ==================================
# Build Network
# ==================================
activation = "tanh"

nNeurons = 24

inputs = keras.Input(shape=(6,))
x = LN.LyapunovDense(nNeurons)(inputs)
x = LN.LyapunovDense(nNeurons)(x)
outputs = Dot(axes=1)([x, x])

V_phi = keras.Model(inputs=inputs, outputs=outputs)

print('V_phi: {}'.format(V_phi.summary()))


saveflag = 'LyapunovNetwork_{}n'.format(nNeurons)
print('\nRUNNING PROGRAM USING SAVE FLAG     {}\n'.format(saveflag))


# ==================================
# Load Data
# ==================================

# Load data
print("Loading mat file")
# base_data_folder = 'E:/Research_Data/StabilityAnalysis/'
base_data_folder = '/orange/rcstudents/omkarmulekar/StabilityAnalysis/'
formulation = 'pm_3dof_energy/'
matfile = loadmat(base_data_folder+formulation+'ANN2_data.mat')



X_train = matfile['Xtrain2'].reshape(-1,6)
t_train = matfile['ttrain2']






# ==================================
# Training Settings
# ==================================
batch_size = 1000
# epochs = 0
epochs = 10000
patience = 20
wait = 0
best = float('inf')

opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)



# ==================================
# Training Loop
# ==================================

history = {'train_loss': [], 'val_loss': [], 'train_Vdot_history': [], 'val_Vdot_history': []}

# Reserve 10,000 samples for validation.
x_val = X_train[-100000:]
y_val = t_train[-100000:]
x_train = X_train[:-100000]
y_train = t_train[:-100000]

# x_train = X_train[:3000000]
# y_train = t_train[:3000000]



print("x_val: {}".format(x_val.shape))
print("x_train: {}".format(x_train.shape))
print('Training on {} datapoints'.format(x_train.shape[0]))

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(x_val.shape[0])




for epoch in range(epochs):
    
    print("\nStart of epoch {} of {}".format(epoch,epochs))
    
    start_time = time.time()


    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):        
        x_batch_train = tf.cast(x_batch_train, dtype=tf.float32)
        y_batch_train = tf.cast(y_batch_train, dtype=tf.float32)

        with tf.GradientTape() as tape_phi:

            with tf.GradientTape() as tape_x:
                tape_x.watch(x_batch_train)
                V_pred = V_phi(x_batch_train)
            
            dVphi_dx = tape_x.gradient(V_pred, x_batch_train)
                
            Vdotphi = dVphi_dx[:,0]*x_batch_train[:,3] + dVphi_dx[:,1]*x_batch_train[:,4] + dVphi_dx[:,2]*x_batch_train[:,5] + dVphi_dx[:,3]*y_batch_train[:,0] + dVphi_dx[:,4]*y_batch_train[:,1] + dVphi_dx[:,5]*(y_batch_train[:,2] - g)

            train_loss = tf.reduce_max(Vdotphi)

        if step % 1000 == 0:
            print("Epoch {} | train step {} of {}".format(epoch, step,x_train.shape[0]/batch_size))
            print("\tMax Vdotphi: {}".format(Vdotphi.numpy().max()))
            print("\ttrain_loss:  {}".format(train_loss))

        grad = tape_phi.gradient(train_loss, V_phi.trainable_variables)
        opt.apply_gradients(zip(grad, V_phi.trainable_variables))

    history['train_Vdot_history'].append(Vdotphi.numpy().mean())
    history['train_loss'].append(train_loss)
    
    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:      

        with tf.GradientTape() as tape_x:
            tape_x.watch(x_batch_val)
            V_pred = V_phi(x_batch_val)
        
        dVphi_dx = tape_x.gradient(V_pred, x_batch_val)
            
        Vdotphi_val = dVphi_dx[:,0]*x_batch_val[:,3] + dVphi_dx[:,1]*x_batch_val[:,4] + dVphi_dx[:,2]*x_batch_val[:,5] + dVphi_dx[:,3]*y_batch_val[:,0] + dVphi_dx[:,4]*y_batch_val[:,1] + dVphi_dx[:,5]*(y_batch_val[:,2] - g)

        val_loss = tf.reduce_max(Vdotphi_val)


    history['val_Vdot_history'].append(Vdotphi_val.numpy().mean())
    history['val_loss'].append(val_loss)

    print('Val loss:     {}'.format(val_loss))
    print('Val Max Vdot: {}'.format(Vdotphi_val.numpy().max()))

    wait += 1
    print("wait = {}".format(wait))
    if val_loss < best:
        print("\tVal loss: {} < best: {}, setting wait to 0".format(val_loss, best))
        best = val_loss
        wait = 0
    if wait >= patience:
        print("\nEarly Stopping condition met, breaking")
        break


test_V_phi = V_phi.predict(np.array([[1,1,1,1,1,1]]))
print('test_V_phi: {}'.format(test_V_phi))

# Save model
print("\nSaving ANN!")
saveout_filename = base_data_folder + formulation + "NetworkTraining/{}.h5".format(saveflag)
print('Filename: ' + saveout_filename)
V_phi.save(saveout_filename)

print('Plotting')
plt.figure(1)
plt.plot(history['train_Vdot_history'])
plt.plot(history['val_Vdot_history'])
plt.legend(['train', 'validation'], loc='best')
plt.title('Vdot')
# plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Vdot (-)')
plt.savefig('{}nettrain_Vdot.png'.format(saveflag))

plt.figure(2)
plt.plot(history['train_loss'])
plt.plot(history['val_loss'])
plt.legend(['train', 'validation'], loc='best')
plt.title('Loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss (-)')
plt.savefig('{}nettrain_logloss.png'.format(saveflag))
