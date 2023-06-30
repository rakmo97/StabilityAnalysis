from tensorflow import keras
from tensorflow.keras import models
import tensorflow as tf
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import broyden1
from scipy.optimize import broyden2
from scipy.optimize import root
import LyapunovNetwork as LN
from matplotlib import pyplot as plt
from scipy.io import loadmat

#%% ============================================================================
# Function definitions
# ============================================================================
def VdotFunc(xi,u):
    g = 9.81

    with tf.GradientTape() as tape:
        tape.watch(xi)
        V_pred = V_phi(xi)
        
    
    dVdx = tape.gradient(V_pred, xi)
    dVdx = tf.reshape(dVdx, (6))

    xdot = np.array([xi[:,3], xi[:,4], xi[:,5], u[:,0], u[:,1], u[:,2]-g]).reshape(-1)
    xdot = tf.constant(xdot, dtype=tf.float32)
    
    Vdotval = tf.tensordot(dVdx, xdot, axes=1)

    return Vdotval


#%% ============================================================================
# Load Policy
# ============================================================================

# base_data_folder = 'E:/Research_Data/StabilityAnalysis/'
base_data_folder = '/orange/rcstudents/omkarmulekar/StabilityAnalysis/'
formulation = 'pm_3dof_energy/'

# Load data
print("Loading mat file")
matfile = loadmat(base_data_folder+formulation+'ANN2_data.mat')
X_train = matfile['Xtrain2'].reshape(-1,6)
t_train = matfile['ttrain2']


# Load policy
print('Loading Policy')
policy_filename = base_data_folder+formulation+'NetworkTraining/customANN2_703_tanh_n100.h5'
policy = models.load_model(policy_filename)

# Load lyapunov network
# lyapunov_filename = base_data_folder + formulation + "NetworkTraining/LyapunovNetwork_120n.h5"
lyapunov_filename = base_data_folder + formulation + "NetworkTraining/LyapunovNetwork_24n.h5"
V_phi = models.load_model(lyapunov_filename,  custom_objects={'LyapunovDense': LN.LyapunovDense})
test_V_phi = V_phi.predict(np.array([[1,1,1,1,1,1]]))
print('test_V_phi: {}'.format(test_V_phi))

# Save flag
saveflag = 'LyapunovNetwork_Agg_uniform_24n'


# System Dynamics settings
nState    =   6
nCtrl     =   3
g = 9.81


#%% ============================================================================
# Training loop
# ============================================================================

# Set up training/validation split
x_val = X_train[-100000:]
y_val = t_train[-100000:]
x_train = X_train[:-10000]
y_train = t_train[:-10000]

# Training Loop Settings
episodes = 50
maxsteps = 2000
pos_to_aggregate = 100000
epochs = 300

# Batch size
batch_size = 1000

# Bounds for random state generation
lower = [-5, -5,   0, -5, -5, -5]
upper = [ 5,  5,  20,  5,  5,  5]


for episode in range(episodes):
    print("\nStarting Episode {} of {}".format(episode+1,episodes))

    ################## Loop to find positive Vdots #################
    print("Running MC to find positive Vdots")
    # Randomly Generate states for MC
    X_list = np.random.uniform(low=lower,high=upper,size=(pos_to_aggregate,nState))
    # X_list = np.random.randn(pos_to_aggregate,nState)*np.array([3,3,10,3,3,3])
    X_list_tensor = tf.constant(X_list)

    # Predict control input using policy (pi_theta)
    u_test = policy(X_list_tensor)
    u_test = tf.clip_by_value(u_test, (-20,-20,0),(20,20,20))

    # Calculate gradient (dVdx)
    with tf.GradientTape() as tape:
        tape.watch(X_list_tensor)
        V_pred = V_phi(X_list_tensor)

    dVdx = tape.batch_jacobian(V_pred, X_list_tensor)
    dVdx = tf.reshape(dVdx, [-1,6])

    # Calculate xdot and Vdot=dVdx'*xdot
    xdot = np.array([X_list[:,3], X_list[:,4], X_list[:,5], u_test[:,0], u_test[:,1], u_test[:,2]-g]).T
    xdot = tf.constant(xdot, dtype=tf.float64)
    Vdot = tf.einsum('ij, ij->i', dVdx, xdot).numpy()

    # Find positive and negative Vdot indices
    negative_idx = np.argwhere(Vdot <= 0)
    positive_idx = np.argwhere(Vdot > 0)

    Vdot_negative = Vdot[negative_idx]
    Vdot_positive = Vdot[positive_idx]


    print('# Positive Vdots: {}'.format(positive_idx.shape[0]))
    print('% Positive Vdots: {} %'.format(100*positive_idx.shape[0]/pos_to_aggregate))
    print('# Negative Vdots: {}'.format(negative_idx.shape[0]))
    print('% Negative Vdots: {} %'.format(100*negative_idx.shape[0]/pos_to_aggregate))

    # Algorithm termination condition (no positive Vdots)
    if positive_idx.shape[0] == 0:
        print('\nNo positive Vdots found for {} random states'.format(pos_to_aggregate))
        break

    # Combine x's from positive Vdots with X_train
    x_train = np.vstack((x_train, X_list))
    y_train = np.vstack((y_train, policy.predict(X_list)))


    ################## Loop to retrain V_phi #################

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(x_val.shape[0])

    # Define optimizer
    opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)

    # Early Stopping conditions
    patience = 8
    wait = 0
    best = float('inf')

    # Training loop
    print("Retraining V_phi")
    for epoch in range(epochs):
        print("Starting training epoch {} of {}".format(epoch+1,epochs))
        if epoch+1 == epochs:
            print("Last training epoch, did not hit early stopping condition (yet)")

        # Training data loop
        printed_this_epoch = False
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):        
            x_batch_train = tf.cast(x_batch_train, dtype=tf.float32)
            y_batch_train = policy(x_batch_train)
            y_batch_train = tf.clip_by_value(y_batch_train, (-20,-20,0),(20,20,20))

            with tf.GradientTape() as tape_phi:

                with tf.GradientTape() as tape_x:
                    tape_x.watch(x_batch_train)
                    V_pred = V_phi(x_batch_train)
                
                dVphi_dx = tape_x.gradient(V_pred, x_batch_train)
                    
                Vdotphi = dVphi_dx[:,0]*x_batch_train[:,3] + dVphi_dx[:,1]*x_batch_train[:,4] + dVphi_dx[:,2]*x_batch_train[:,5] + dVphi_dx[:,3]*y_batch_train[:,0] + dVphi_dx[:,4]*y_batch_train[:,1] + dVphi_dx[:,5]*(y_batch_train[:,2] - g)
                
                train_loss = tf.reduce_max(Vdotphi)
                # train_loss = tf.reduce_max(Vdotphi) + tf.reduce_mean(tf.norm(dVphi_dx, axis=1))

            if Vdotphi.numpy().max() > 0. and not printed_this_epoch:
                print("\tEpisode {}, Epoch {} | train step {} of {}".format(episode+1,epoch+1, step,x_train.shape[0]/batch_size))
                print("\tMax Vdotphi: {}".format(Vdotphi.numpy().max()))
                print("\ttrain_loss:  {}".format(train_loss))
                printed_this_epoch = True

            grad = tape_phi.gradient(train_loss, V_phi.trainable_variables)
            opt.apply_gradients(zip(grad, V_phi.trainable_variables))

        # Validation data loop
        for x_batch_val, t_batch_val in val_dataset:      
            x_batch_val = tf.cast(x_batch_val, dtype=tf.float32)
            y_batch_val = policy(x_batch_val)

            with tf.GradientTape() as tape_x:
                tape_x.watch(x_batch_val)
                V_pred = V_phi(x_batch_val)
            
            dVphi_dx = tape_x.gradient(V_pred, x_batch_val)
            
            Vdotphi_val = dVphi_dx[:,0]*x_batch_val[:,3] + dVphi_dx[:,1]*x_batch_val[:,4] + dVphi_dx[:,2]*x_batch_val[:,5] + dVphi_dx[:,3]*y_batch_val[:,0] + dVphi_dx[:,4]*y_batch_val[:,1] + dVphi_dx[:,5]*(y_batch_val[:,2] - g)
        
            val_loss = tf.reduce_max(Vdotphi_val)
            # val_loss = tf.reduce_max(Vdotphi_val) + tf.reduce_mean(tf.norm(dVphi_dx, axis=1))

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



print("\nSaving ANN!")
saveout_filename = base_data_folder + formulation + "NetworkTraining/{}.h5".format(saveflag)
print('Filename: ' + saveout_filename)
V_phi.save(saveout_filename)
