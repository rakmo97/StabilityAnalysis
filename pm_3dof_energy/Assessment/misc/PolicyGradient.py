from tensorflow.keras import models
import tensorflow as tf
import numpy as np
from scipy.optimize import fsolve


#%% ============================================================================
# Load Policy
# ============================================================================
print('Loading Policy')
base_data_folder = '/orange/rcstudents/omkarmulekar/3DoF_RigidBody/'
filename = base_data_folder+'Code_coupled_constmass_normed/NetworkTraining/ANN2_703_tanh_n150.h5'
policy = models.load_model(filename)

nState    =   6
nCtrl     =   2


#%% ============================================================================
# Load Policy
# ============================================================================
x_Vdotmax = tf.convert_to_tensor(np.array([-14.41407121, 7.11345874, -14.73887945, 0, 0, 0]).reshape(1,-1), dtype=tf.float32)

with tf.GradientTape() as t2:
    print('Inside GradientTape')
    t2.watch(x_Vdotmax)
    print('Tape Watching')
    
    with tf.GradientTape() as t1:
        t1.watch(x_Vdotmax)
        u = policy(x_Vdotmax)
    g = t1.jacobian(u, x_Vdotmax)


print('Forward Pass Complete')
print('u = {}'.format(u))

h = t2.jacobian(g, x_Vdotmax).numpy().reshape(2,6,6)
g = g.numpy().reshape(2,6)

print('Gradient = {}'.format(g))
print('Gradient Shape: {}'.format(g.shape))
print('Hessian = {}'.format(h))
print('Hessian Shape: {}'.format(h.shape))
