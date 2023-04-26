from tensorflow import keras
import tensorflow as tf

import numpy as np
from LyapunovNetwork import LyapunovNetwork
from tensorflow.keras import models
from scipy.io import loadmat
from matplotlib import pyplot as plt
import LanderDynamics as LD
import time




V_phi = LyapunovNetwork()


 # Load data
print("Loading mat file")
base_data_folder = 'E:/Research_Data/StabilityAnalysis/'
# base_data_folder = '/orange/rcstudents/omkarmulekar/StabilityAnalysis/'
formulation = 'pm_3dof/'
matfile = loadmat(base_data_folder+formulation+'ANN2_data.mat')
saveflag = 'customANN2'
# saveflag = 'fullmin_max'
# saveflag = 'fullmin_max1step'
# saveflag = 'fullmin_max1step_20episodes'


Xfull = matfile['Xfull_2']
tfull = matfile['tfull_2']
X_train = matfile['Xtrain2'].reshape(-1,6)
t_train = matfile['ttrain2']
X_test = matfile['Xtest2'].reshape(-1,6)
t_test = matfile['ttest2']


X_test = X_test[:10000,:]

# Load trained policy
print('Loading Policy')
filename = base_data_folder+formulation+'NetworkTraining/customANN2_703_tanh_n100.h5'
# filename = base_data_folder+formulation+'NetworkTraining/fullmin_max_ANN2_703_tanh_n100.h5'
# filename = base_data_folder+formulation+'NetworkTraining/fullmin_max1step_ANN2_703_tanh_n100.h5'
# filename = base_data_folder+formulation+'NetworkTraining/fullmin_max1step_20episodesANN2_703_tanh_n100.h5'
policy = models.load_model(filename)

nState    =   6
nCtrl     =   3
g = 9.81




batch_size = 10000
epochs = 10000
patience = 25
wait = 0
best = float('inf')


opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)



# ==================================
# Training Loop
# ==================================

history = {'Vdot_history': [], 'Vdot_val_history': []}

# Reserve 10,000 samples for validation.
x_val = X_train[-10000:]
y_val = t_train[-10000:]
# x_train = X_train[:-10000]
# y_train = t_train[:-10000]

x_train = X_train[:-4500000]
y_train = t_train[:-4500000]

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
    
    print("Start of epoch {} of {}".format(epoch,epochs))
    
    start_time = time.time()


    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        
        u_pred = policy.predict(x_batch_train)
        u_pred[:,0] = np.clip(u_pred[:,0], -20, 20)
        u_pred[:,1] = np.clip(u_pred[:,1], -20, 20)
        u_pred[:,2] = np.clip(u_pred[:,2],   0, 20)
        u_pred = tf.cast(u_pred, dtype=tf.float64)

        
        with tf.GradientTape() as tape_phi:

            with tf.GradientTape() as tape_x:
                tape_x.watch(x_batch_train)
                V_pred = V_phi(x_batch_train)
            
            dVphi_dx = tape_x.gradient(V_pred, x_batch_train)
        
            
            Vdotphi = tf.reduce_mean(dVphi_dx[:,0]*x_batch_train[:,3] + dVphi_dx[:,1]*x_batch_train[:,4] + dVphi_dx[:,2]*x_batch_train[:,5] + dVphi_dx[:,3]*u_pred[:,0] + dVphi_dx[:,4]*u_pred[:,1] + dVphi_dx[:,5]*(u_pred[:,2] - g))

        if step % 10 == 0:
            print("Vdotphi for step {} of {}: {}".format(step,x_train.shape[0]/batch_size,Vdotphi))

        dVdotphi_dphi = tape_phi.gradient(Vdotphi, V_phi.trainable_variables)
        opt.apply_gradients(zip(dVdotphi_dphi, V_phi.trainable_variables))

    history['Vdot_history'].append(Vdotphi.numpy())
    
    
    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        u_pred = policy.predict(x_batch_val)
        u_pred[:,0] = np.clip(u_pred[:,0], -20, 20)
        u_pred[:,1] = np.clip(u_pred[:,1], -20, 20)
        u_pred[:,2] = np.clip(u_pred[:,2],   0, 20)
        u_pred = tf.cast(u_pred, dtype=tf.float64)

        

        with tf.GradientTape() as tape_x:
            tape_x.watch(x_batch_val)
            V_pred = V_phi(x_batch_val)
        
        dVphi_dx = tape_x.gradient(V_pred, x_batch_val)
    
        
        Vdotphi_val = tf.reduce_mean(dVphi_dx[:,0]*x_batch_val[:,3] + dVphi_dx[:,1]*x_batch_val[:,4] + dVphi_dx[:,2]*x_batch_val[:,5] + dVphi_dx[:,3]*u_pred[:,0] + dVphi_dx[:,4]*u_pred[:,1] + dVphi_dx[:,5]*(u_pred[:,2] - g))
   
    history['Vdot_val_history'].append(Vdotphi_val.numpy())

    wait += 1
    if Vdotphi_val < best:
        print("Val loss: {} < best: {}, setting wait to 0".format(Vdotphi_val, best))
        best = Vdotphi_val
        wait = 0
    if wait >= patience:
        print("Early Stopping condition met, breaking")
        break



# Save model
print("\nSaving ANN!")
saveout_filename = base_data_folder + formulation + "NetworkTraining/MinimizedLyapunovNetwork"
print('Filename: ' + saveout_filename)
V_phi.save_weights(saveout_filename)

plt.figure(1)
plt.plot(history['Vdot_history'])
plt.plot(history['Vdot_val_history'])
plt.legend(['train', 'validation'], loc='best')
plt.title('Vdot')
# plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Vdot (-)')
plt.savefig('{}nettrain_Vdot.png'.format(saveflag))
