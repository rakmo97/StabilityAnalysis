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
# Load Policy
# ============================================================================
print('Loading Policy')
# base_data_folder = 'E:/Research_Data/StabilityAnalysis/'
base_data_folder = '/orange/rcstudents/omkarmulekar/StabilityAnalysis/'
formulation = 'pm_3dof/'
policy_filename = base_data_folder+formulation+'NetworkTraining/trainThetaPhi_MCLyapunov_ANN2_703_tanh_n250.h5'

policy = models.load_model(policy_filename)


lyapunov_filename = base_data_folder + formulation + "NetworkTraining/trainThetaPhi_MCLyapunov__Lyappunov.h5"

# Create an instance of the model
V_phi = models.load_model(lyapunov_filename,  custom_objects={'LyapunovDense': LN.LyapunovDense})
test_V_phi = V_phi.predict(np.array([[1,1,1,1,1,1]]))
print('test_V_phi: {}'.format(test_V_phi))


saveflag = 'Max_Basic'

nState    =   6
nCtrl     =   3

plotting = False

#%% ============================================================================
# Load Data
# ============================================================================
matfile = loadmat(base_data_folder+formulation+'ANN2_data.mat')
saveflag = 'trainThetaPhi_MCLyapunov_10break_normal_'
print('\nRUNNING PROGRAM USING SAVE FLAG     {}\n'.format(saveflag))

Xfull = matfile['Xfull_2']
tfull = matfile['tfull_2']
X_train = matfile['Xtrain2'].reshape(-1,6)
t_train = matfile['ttrain2']
X_test = matfile['Xtest2'].reshape(-1,6)
t_test = matfile['ttest2']


#%% ============================================================================
# Test eigenvalue gradient calc
# ============================================================================

x = tf.constant(X_train[:100,:], dtype=tf.float32)
g = tf.constant(9.81, dtype=tf.float32)



with tf.GradientTape() as tape_params: # to calculate dEig/dtheta
   
    with tf.GradientTape() as t1: # To calculate d2Vdot/dx2 (hessian of Vdot)
        t1.watch(x)
        
        with tf.GradientTape() as t2: # To calculate dVdot/dx (Gradient of Vdot)
            t2.watch(x)
            
            u = policy(x)
            u = tf.clip_by_value(u, (-20,-20,0),(20,20,20))

            xdot = np.array([x[:,3], x[:,4], x[:,5], u[:,0], u[:,1], u[:,2]-g]).T
            xdot = tf.constant(xdot, dtype=tf.float32)
           
            # Calculate Vdot
            with tf.GradientTape() as t3: # To calculate dV/dx (Gradient of V)
                t3.watch(x)
                V_pred = V_phi(x)

            dVdx = t3.gradient(V_pred, x)
            dVdx = tf.reshape(dVdx, (-1,6))

            Vdot = tf.einsum('ij, ij->i', dVdx, xdot)


        grad_Vdot = t2.jacobian(Vdot, x)
        
    hessian_Vdot_out = t1.jacobian(grad_Vdot, x)
    hessian_Vdot = hessian_Vdot_out.numpy().reshape(6,6)

    eigs = np.linalg.eig(hessian_Vdot)[0]
    max_eig = tf.reduce_max(eigs)

grads = tape_params.gradient(train_loss, policy.trainable_weights)

print('x: {}'.format(x))
print('Vdot: {}'.format(Vdot))
print('grad_Vdot: {}'.format(grad_Vdot))
print('hessian_Vdot: {}'.format(hessian_Vdot))
print('hessian eigenvalues: {}'.format(eigs))
print('hessian eigenvalue max: {}'.format(eigs.max()))