from tensorflow.keras import models
import tensorflow as tf
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import broyden1
from scipy.optimize import broyden2
from scipy.optimize import root
import LyapunovNetwork as LN
from matplotlib import pyplot as plt

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
print('Loading Policy')
# base_data_folder = 'E:/Research_Data/StabilityAnalysis/'
base_data_folder = '/orange/rcstudents/omkarmulekar/StabilityAnalysis/'
formulation = 'pm_3dof/'
# policy_filename = base_data_folder+formulation+'NetworkTraining/customANN2_703_tanh_n100.h5'
policy_filename = base_data_folder+formulation+'NetworkTraining/trainThetaPhi_MCLyapunov_ANN2_703_tanh_n250.h5'
# policy_filename = base_data_folder+formulation+'NetworkTraining/regloss_w5_adam_ANN2_703_tanh_n100.h5'
# policy_filename = base_data_folder+formulation+'NetworkTraining/lagrangianParametricLyapunov_Max_ANN2_703_tanh_n250.h5'

policy = models.load_model(policy_filename)


# lyapunov_filename = base_data_folder + formulation + "NetworkTraining/MinimizedLyapunovNetwork_MSE_lyapunov_MSE_Max.h5"
# lyapunov_filename = base_data_folder + formulation + "NetworkTraining/MinimizedLyapunovNetwork_MSE_lyapunov_MSE_MeanStrong.h5"
# lyapunov_filename = base_data_folder + formulation + "NetworkTraining/MinimizedLyapunovNetwork_MSE_lyapunov_MSE_MeanStrong_batch10.h5"
# lyapunov_filename = base_data_folder + formulation + "NetworkTraining/MinimizedLyapunovNetwork_MSE_lyapunov_MSE_Max_batch5.h5"
# lyapunov_filename = base_data_folder + formulation + "NetworkTraining/MinimizedLyapunovNetwork_MSE_LyBasic_Max.h5"
# lyapunov_filename = base_data_folder + formulation + "NetworkTraining/AggregateLearning_Aggregate_Learning.h5"
# lyapunov_filename = base_data_folder + formulation + "NetworkTraining/AggregateLearningMC_Aggregate_Learning_320n.h5"
# lyapunov_filename = base_data_folder + formulation + "NetworkTraining/MinimizedLyapunovNetwork_MSE_LyBasic_Max_100n.h5"
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
# Hand-coded solver loop using tensorflow
# ============================================================================

epochs = 0
x0 = tf.constant([0.,0.,0.,0.,0.,0.], dtype=tf.float32)
# x0 = tf.constant([-0.38423282, -0.40785056,  0.97273065 , 4.90704575,  2.53847815 , 0.24324055], dtype=tf.float32)
xcurr = x0
xlast = x0


history = {'x': [], 'dVdotdx': [], 'Vdot': []}
history['x'].append(x0.numpy().reshape(1,-1))

# Zero-th step
x = tf.reshape(tf.constant(xcurr, dtype=tf.float32), (1,-1))
with tf.GradientTape() as t2:
    t2.watch(x)
    with tf.GradientTape() as t1:
        t1.watch(x)
        u = policy(x)
        u = tf.clip_by_value(u, (-20,-20,0),(20,20,20))
        Vdot = VdotFunc(x, u)

    grad_Vdot = t1.jacobian(Vdot, x)
hessian_Vdot_out = t2.jacobian(grad_Vdot, x)
hessian_Vdot = hessian_Vdot_out.numpy().reshape(6,6)
eigs = np.linalg.eig(hessian_Vdot)[0]

print('x: {}'.format(x))
print('Vdot: {}'.format(Vdot))
print('grad_Vdot: {}'.format(grad_Vdot))
print('hessian_Vdot: {}'.format(hessian_Vdot))
print('hessian eigenvalues: {}'.format(eigs))
print('hessian eigenvalue max: {}'.format(eigs.max()))

history['dVdotdx'].append(grad_Vdot.numpy())
history['Vdot'].append(Vdot.numpy())
mu = 1.0

print('Starting solver loop')
for epoch in range(epochs):

    print("\nStart of epoch {} of {}".format(epoch,epochs))

    # Calculate gradient [i.e. f(x)] and hessian [i.e. f'(x)]
    x = tf.reshape(tf.constant(xcurr, dtype=tf.float32), (1,-1))
    with tf.GradientTape() as t2:
        t2.watch(x)

        with tf.GradientTape() as t1:
            t1.watch(x)
            u = policy(x)
            u = tf.clip_by_value(u, (-20,-20,0),(20,20,20))

            Vdot = VdotFunc(x, u)
        
        grad_Vdot = t1.jacobian(Vdot, x) # this is f(x)

    hessian_Vdot = t2.jacobian(grad_Vdot, x).numpy().reshape(6,6) # this is f'(x)

    print('x: {}'.format(x))
    print('grad_Vdot: {}'.format(grad_Vdot))
    print('Vdot: {}'.format(Vdot))
    print('Max Hessian eigenvalue: {}'.format(np.linalg.eig(hessian_Vdot)[0].max()))

    # Newton: x[n+1] = x[n] - inv(f'(x[n]))*f(x[n])
    # xcurr = tf.reshape(xlast - tf.transpose( tf.linalg.inv(hessian_Vdot) @ tf.transpose(grad_Vdot)), (nState))

    # Levenberg-Marquardt: x[n+1] = x[n] - inv(f'(x[n]).T * f'(x[n]) + mu[k]*I)*f(x[n])
    # xcurr = tf.reshape(xlast - tf.transpose( tf.linalg.inv(tf.transpose(hessian_Vdot)@tf.transpose(hessian_Vdot)+mu*tf.eye(nState)) @ tf.transpose(grad_Vdot) ) , (nState))
    
    # Gradient Ascent: x[n+1] = x[n] + eta*f(x[n])
    xcurr = xlast + 0.0001*grad_Vdot

    xcurr = tf.clip_by_value(xcurr, (-5, -5, -20, -5, -5, -5), (5, 5, 20, 5, 5, 5))

    history['x'].append(xcurr.numpy())
    history['dVdotdx'].append(grad_Vdot.numpy())
    history['Vdot'].append(Vdot.numpy())

    if np.isclose(grad_Vdot, int(nState)*[0.0]).all(): # if f(x[n]) near zero
        break

    if np.isclose(abs(xcurr - xlast),0.0).all(): # if x is converging
        break

    xlast = xcurr

print('xlast: {}'.format(xlast))

#%% ============================================================================
# Post Processing
# ============================================================================
Vdot_all = np.array(history['Vdot'])
negative_idx = np.argwhere(Vdot_all <= 0).reshape(-1)
positive_idx = np.argwhere(Vdot_all > 0).reshape(-1)

X_all = np.concatenate(history['x'],axis=0)
X_positive = X_all[positive_idx]
X_negative = X_all[negative_idx]


#%% ============================================================================
# Plotting
# ============================================================================
if plotting:
    plt.figure(1)
    plt.plot(history['Vdot'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Vdot (-)')
    plt.savefig('{}Vdot.png'.format(saveflag))


    plt.figure(2)
    plt.plot(X_negative[:,0],X_negative[:,3])
    plt.plot(X_positive[:,0],X_positive[:,3])
    plt.legend(['Vdot<=0', 'Vdot>0'], loc='best')
    plt.title('Phasespace X')
    plt.xlabel('x [m]')
    plt.ylabel('vx [m/s]')
    plt.savefig('{}_phasespace_x.png'.format(saveflag))


    plt.figure(3)
    plt.plot(X_negative[:,1],X_negative[:,4])
    plt.plot(X_positive[:,1],X_positive[:,4])
    plt.legend(['Vdot<=0', 'Vdot>0'], loc='best')
    plt.title('Phasespace Y')
    plt.xlabel('y [m]')
    plt.ylabel('vy [m/s]')
    plt.savefig('{}_phasespace_y.png'.format(saveflag))


    plt.figure(4)
    plt.plot(X_negative[:,2],X_negative[:,5])
    plt.plot(X_positive[:,2],X_positive[:,5])
    plt.legend(['Vdot<=0', 'Vdot>0'], loc='best')
    plt.title('Phasespace Z')
    plt.xlabel('z [m]')
    plt.ylabel('vz [m/s]')
    plt.savefig('{}_phasespace_z.png'.format(saveflag))

