from tensorflow.keras import models
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt



#%% ============================================================================
# Load Policy
# ============================================================================

print('Loading Policy')
# base_data_folder = 'E:/Research_Data/StabilityAnalysis/'
base_data_folder = '/orange/rcstudents/omkarmulekar/StabilityAnalysis/'
formulation = 'pm_3dof/'
policy_filename = base_data_folder+formulation+'NetworkTraining/customANN2_703_tanh_n100.h5'

policy = models.load_model(policy_filename)


nState    =   6
nCtrl     =   3
g = 9.81

#%% ============================================================================
# Calculate Gradients
# ============================================================================

x = tf.constant([[0.,0.,0.,0.,0.,0.]], dtype=tf.float32) # Equilibrium point (origin)

# Calculate du/dx
with tf.GradientTape() as tape:
    tape.watch(x)
    u = policy(x)
    u = tf.clip_by_value(u, (-20,-20,0),(20,20,20))

du_dx = tape.jacobian(u,x).numpy().reshape(nCtrl,nState)

df_du = np.array([[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.],
                  [1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]])

df_dx = np.array([[0., 0., 0., 1., 0., 0.],
                  [0., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 1.],
                  [0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0.]])



# Construct A_cl

A_cl = df_dx + tf.matmul(df_du, du_dx)
eig = np.linalg.eig(A_cl)[0]
eig_max = eig.max()

print('A_cl: {}'.format(A_cl))
print('Eigenvalues: {}'.format(eig))
print('Max Eig: {}'.format(eig_max))