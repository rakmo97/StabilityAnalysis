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
formulation = 'pm_3dof/'

# Load data
print("Loading mat file")
matfile = loadmat(base_data_folder+formulation+'ANN2_data.mat')
X_train = matfile['Xtrain2'].reshape(-1,6)
t_train = matfile['ttrain2']


# Load policy
print('Loading Policy')
policy_filename = base_data_folder+formulation+'NetworkTraining/lagrangianParametricLyapunov_Max_ANN2_703_tanh_n250.h5'
policy = models.load_model(policy_filename)

# Load lyapunov network
lyapunov_filename = base_data_folder + formulation + "NetworkTraining/MinimizedLyapunovNetwork_MSE_LyBasic_Max.h5"
V_phi = models.load_model(lyapunov_filename,  custom_objects={'LyapunovDense': LN.LyapunovDense})
test_V_phi = V_phi.predict(np.array([[1,1,1,1,1,1]]))
print('test_V_phi: {}'.format(test_V_phi))


saveflag = 'Aggregate_Learning'


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

# Training Loop
episodes = 10
maxsteps = 2000
pos_to_aggregate = 1000
epochs = 300



for episode in range(episodes):
    print("Starting Episode {} of {}".format(episode+1,episodes))

    # Loop to find positive Vdots
    print("Finding positive Vdots")
    x0 = tf.constant([0.,0.,0.,0.,0.,0.], dtype=tf.float32)
    xcurr = x0
    xlast = x0
    x_pos_list = np.empty((0,6))
    for maxstep in range(maxsteps):
        if maxstep % 500 == 0:
            print("Starting maxstep {} of {}".format(maxstep+1,maxsteps))

        # Calculate gradient [i.e. f(x)] and hessian [i.e. f'(x)]
        x = tf.reshape(tf.constant(xcurr, dtype=tf.float32), (1,-1))

        with tf.GradientTape() as t1:
            t1.watch(x)
            u = policy(x)
            Vdot = VdotFunc(x, u)
        
        dVdot_dx = t1.jacobian(Vdot, x) # this is f(x)
      
        xcurr = xlast + 0.01*dVdot_dx

        if Vdot > 0.0:
            x_pos_list = np.vstack((x_pos_list, xcurr.numpy()))
        
        if x_pos_list.shape[0] == pos_to_aggregate:
            print("Got {} positive Vdots, breaking loop and continuing to Retrain step".format(pos_to_aggregate))
            break

        xlast = xcurr

    # Combine x's from positive Vdots with X_train
    x_train = np.vstack((x_train, x_pos_list))
    y_train = np.vstack((y_train, np.zeros((x_pos_list.shape[0],nCtrl))))



    # Batch size
    batch_size = 1000

    # Early Stopping conditions
    patience = 8
    wait = 0
    best = float('inf')


    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(x_val.shape[0])

    # Define optimizer
    opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)


    # Loop to retrain V_phi
    print("Retraining V_phi")
    for epoch in range(epochs):
        print("Starting training epoch {} of {}".format(epoch+1,epochs))
        
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):        
            x_batch_train = tf.cast(x_batch_train, dtype=tf.float32)
            y_batch_train = policy(x_batch_train)



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


        for x_batch_val, t_batch_val in val_dataset:      
            x_batch_val = tf.cast(x_batch_val, dtype=tf.float32)
            y_batch_val = policy(x_batch_val)

            with tf.GradientTape() as tape_x:
                tape_x.watch(x_batch_val)
                V_pred = V_phi(x_batch_val)
            
            dVphi_dx = tape_x.gradient(V_pred, x_batch_val)
            
            Vdotphi_val = dVphi_dx[:,0]*x_batch_val[:,3] + dVphi_dx[:,1]*x_batch_val[:,4] + dVphi_dx[:,2]*x_batch_val[:,5] + dVphi_dx[:,3]*y_batch_val[:,0] + dVphi_dx[:,4]*y_batch_val[:,1] + dVphi_dx[:,5]*(y_batch_val[:,2] - g)
        
            val_loss = tf.reduce_max(Vdotphi_val)

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
saveout_filename = base_data_folder + formulation + "NetworkTraining/AggregateLearning_{}.h5".format(saveflag)
print('Filename: ' + saveout_filename)
V_phi.save(saveout_filename)