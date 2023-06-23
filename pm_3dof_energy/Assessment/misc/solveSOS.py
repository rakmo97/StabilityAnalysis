from tensorflow.keras import models
import tensorflow as tf
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import broyden1
from scipy.optimize import broyden2
from scipy.optimize import root
import LyapunovNetwork as LN


#%% ============================================================================
# Load Policy
# ============================================================================
print('Loading Policy')
# base_data_folder = 'E:/Research_Data/StabilityAnalysis/'
base_data_folder = '/orange/rcstudents/omkarmulekar/StabilityAnalysis/'
formulation = 'pm_3dof/'
# filename = base_data_folder+formulation+'NetworkTraining/ANN2_703_tanh_n100.h5'
filename = base_data_folder+formulation+'NetworkTraining/customANN2_703_tanh_n100.h5'

policy = models.load_model(filename)


# saveout_filename = base_data_folder + formulation + "NetworkTraining/MinimizedLyapunovNetwork_MSE_lyapunov_MSE_Max.h5"
# saveout_filename = base_data_folder + formulation + "NetworkTraining/MinimizedLyapunovNetwork_MSE_lyapunov_MSE_MeanStrong.h5"
# saveout_filename = base_data_folder + formulation + "NetworkTraining/MinimizedLyapunovNetwork_MSE_lyapunov_MSE_MeanStrong_batch10.h5"
# saveout_filename = base_data_folder + formulation + "NetworkTraining/MinimizedLyapunovNetwork_MSE_lyapunov_MSE_Max_batch5.h5"
saveout_filename = base_data_folder + formulation + "NetworkTraining/MinimizedLyapunovNetwork_MSE_LyBasic_Max.h5"

# Create an instance of the model
V_phi = models.load_model(saveout_filename,  custom_objects={'LyapunovDense': LN.LyapunovDense})
test_V_phi = V_phi.predict(np.array([[1,1,1,1,1,1]]))
print('test_V_phi: {}'.format(test_V_phi))


nState    =   6
nCtrl     =   3

#%% ============================================================================
# Construct Fsolve Function
# ============================================================================

print('Constructing solver function')

# @tf.function
def func(x, policy):
    x = x.reshape(1,-1)
    x = tf.constant(x, dtype=tf.float32)


    with tf.GradientTape() as tape:
        tape.watch(x)
        u = policy(x)
        Vdotval = Vdot(x,u)
    
    dVdot_dx = tape.jacobian(Vdotval,x)
    dVdot_dx = tf.reshape(dVdot_dx, (6))

    return dVdot_dx



def Vdot(xi,u):
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


def CalculateHessian(xi, policy):
    xi = tf.convert_to_tensor(xi.reshape(1,-1), dtype=tf.float32)
    
    with tf.GradientTape() as t2:
        t2.watch(xi)
        
        with tf.GradientTape() as t1:
            t1.watch(xi)
            u = policy(xi)
        
        g = t1.jacobian(u, xi)
    
    h = t2.jacobian(g, xi).numpy().reshape(3,6,6)
    # print('Policy H: {}'.format(h))

    return h



#%% ============================================================================
# Call Fsolve to solve system of equations
# ============================================================================
print('Calling Solver')

# x0 = [0.1,0.1,0.1,0.1,0.1,0.1]
x0 = [1.,1.,1.,1.,1.,1.]
x_Vdotmax = fsolve(func, x0, args=policy)

print('Solution: {}'.format(x_Vdotmax))

x_Vdotmax_tensor = tf.reshape(tf.constant(x_Vdotmax, dtype=tf.float32), (1,-1))

with tf.GradientTape() as t2:
    t2.watch(x_Vdotmax_tensor)

    with tf.GradientTape() as t1:
        t1.watch(x_Vdotmax_tensor)
        u_Vdotmax = policy(x_Vdotmax_tensor)
        Vdotmax = Vdot(x_Vdotmax_tensor, u_Vdotmax)
    
    grad_Vdot = t1.jacobian(Vdotmax, x_Vdotmax_tensor)

hessian_Vdot = t2.jacobian(grad_Vdot, x_Vdotmax_tensor).numpy().reshape(6,6)

print('========== Solution =========')
print('x_Vdotmax = {}'.format(x_Vdotmax))
print('Vdotmax = {}'.format(Vdotmax))
print('grad_Vdot = {}'.format(grad_Vdot))
print('hessian_Vdot = {}'.format(hessian_Vdot))

eig = np.linalg.eig(hessian_Vdot)
eigvals = eig[0]
eigvecs = eig[1]

print('Max eig value: {}'.format(eigvals.max()))
