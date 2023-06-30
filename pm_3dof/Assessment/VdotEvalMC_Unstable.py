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
# Load Policy
# ============================================================================
print('Loading Policy')
# base_data_folder = 'E:/Research_Data/StabilityAnalysis/'
base_data_folder = '/orange/rcstudents/omkarmulekar/StabilityAnalysis/'
formulation = 'pm_3dof/'
# # policy_filename = base_data_folder+formulation+'NetworkTraining/customANN2_703_tanh_n100.h5'
# policy_filename = base_data_folder+formulation+'NetworkTraining/trainThetaPhi_MCLyapunov_grad__policy.h5'

# policy = models.load_model(policy_filename)
policy = lambda x : tf.constant([[20.,20.,20.]]*x.shape[0])


# Load Lyapunov Network
lyapunov_filename = base_data_folder + formulation + "NetworkTraining/AggregateLearningMC_Aggregate_Learning_Unstable.h5"

V_phi = models.load_model(lyapunov_filename,  custom_objects={'LyapunovDense': LN.LyapunovDense})
test_V_phi = V_phi(np.array([[1,1,1,1,1,1]])).numpy()
print('test_V_phi: {}'.format(test_V_phi))


saveflag = 'MC_eval_Unstable'

nState    =   6
nCtrl     =   3
g = 9.81

plotting = True

#%% ============================================================================
# Monte Carlo (evaluations)
# ============================================================================
lower = [-5, -5,   0, -5, -5, -5]
upper = [ 5,  5,  20,  5,  5,  5]
# lower = [-1, -1,   0, -1, -1, -1]
# upper = [ 1,  1,  1,  1,  1,  1]
# lower = [2, 3,   3, -1, -1, -1]
# upper = [ 4,  5,  4,  1,  1,  1]

nTest = 3000000

# X_list = np.random.uniform(low=lower,high=upper,size=(nTest,nState))
X_list = np.random.randn(nTest,nState)*np.array([3,3,10,3,3,3])

X_list_tensor = tf.constant(X_list)
print("Calculating u(x)")
u_test = policy(X_list_tensor)
u_test = tf.clip_by_value(u_test, (-20,-20,0),(20,20,20))

with tf.GradientTape() as tape:
    tape.watch(X_list_tensor)
    V_pred = V_phi(X_list_tensor)


dVdx = tape.batch_jacobian(V_pred, X_list_tensor)

dVdx = tf.reshape(dVdx, [-1,6])


xdot = np.array([X_list[:,3], X_list[:,4], X_list[:,5], u_test[:,0], u_test[:,1], u_test[:,2]-g]).T
xdot = tf.constant(xdot, dtype=tf.float64)
Vdot = tf.einsum('ij, ij->i', dVdx, xdot).numpy()


negative_idx = np.argwhere(Vdot <= 0)
positive_idx = np.argwhere(Vdot > 0)


Vdot_negative = Vdot[negative_idx]
Vdot_positive = Vdot[positive_idx]


print('# Positive Vdots: {}'.format(positive_idx.shape[0]))
print('% Positive Vdots: {} %'.format(100*positive_idx.shape[0]/nTest))
print('# Negative Vdots: {}'.format(negative_idx.shape[0]))
print('% Negative Vdots: {} %'.format(100*negative_idx.shape[0]/nTest))
print("Min of Negative Vdots: {}".format(Vdot_negative.min()))
print("Max of Negative Vdots: {}".format(Vdot_negative.max()))
print("Max of Positive Vdots: {}".format(Vdot_positive.max())) if positive_idx.shape[0] != 0 else print("no positive Vdots")


# abs_minz_idx =  np.argmin(abs(X_list[positive_idx.reshape(-1),5]))

# # print("Positive X: {}".format(X_list[positive_idx[0],:]))
# print("Positive X: {}".format(X_list[positive_idx[abs_minz_idx],:]))






if plotting:    
    plt.figure(1)
    plt.plot(Vdot_negative,'.')
    plt.plot(Vdot_positive,'.')
    plt.legend(['Vdot<=0', 'Vdot>0'], loc='best')
    plt.title('Vdot')
    plt.xlabel('Index [-]')
    plt.ylabel('Vdot [-]')
    plt.savefig('{}Vdot.png'.format(saveflag))



    # plt.figure(2)
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(X_list[negative_idx,0], X_list[negative_idx,1], X_list[negative_idx,2])
    # ax.scatter3D(X_list[positive_idx,0],X_list[positive_idx,1],X_list[positive_idx,2])
    # ax.set_xlabel('X [m]')
    # ax.set_ylabel('Y [m]')
    # ax.set_zlabel('Z [m]')
    # ax.legend(['Vdot<=0', 'Vdot>0'], loc='best')
    # plt.savefig('{}_pos.png'.format(saveflag))

    plt.figure(3)
    plt.plot(X_list[negative_idx,0],X_list[negative_idx,3],'.')
    plt.plot(X_list[positive_idx,0],X_list[positive_idx,3],'.')
    # plt.scatter(X_list[:,0],X_list[:,3],s=5,c=Vdot)
    # plt.colorbar()
    plt.legend(['Vdot<=0', 'Vdot>0'], loc='best')
    plt.title('Phasespace X')
    plt.xlabel('x [m]')
    plt.ylabel('vx [m/s]')
    plt.savefig('{}_phasespace_x.png'.format(saveflag))

    plt.figure(4)
    plt.plot(X_list[negative_idx,1],X_list[negative_idx,4],'.')
    plt.plot(X_list[positive_idx,1],X_list[positive_idx,4],'.')
    plt.legend(['Vdot<=0', 'Vdot>0'], loc='best')
    plt.title('Phasespace Y')
    plt.xlabel('y [y]')
    plt.ylabel('vy [m/s]')
    plt.savefig('{}_phasespace_y.png'.format(saveflag))

    plt.figure(5)
    plt.plot(X_list[negative_idx,2],X_list[negative_idx,5],'.')
    plt.plot(X_list[positive_idx,2],X_list[positive_idx,5],'.')
    plt.legend(['Vdot<=0', 'Vdot>0'], loc='best')
    plt.title('Phasespace Z')
    plt.xlabel('z [m]')
    plt.ylabel('vz [m/s]')
    plt.savefig('{}_phasespace_z.png'.format(saveflag))
        