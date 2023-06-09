from scipy.io import loadmat
from scipy.io import savemat
from tensorflow.keras import models
import pickle
import numpy as np
from scipy import integrate
import time
import LanderDynamics as LD

from matplotlib import pyplot as plt


print("Loading mat file")
# base_data_folder = 'E:/Research_Data/StabilityAnalysis/'
base_data_folder = '/orange/rcstudents/omkarmulekar/StabilityAnalysis/'
formulation = 'pm_3dof/'
matfile = loadmat(base_data_folder+formulation+'ANN2_data.mat')
saveflag = 'minP_'


Xfull = matfile['Xfull_2']
tfull = matfile['tfull_2']
X_test = matfile['Xtrain2'].reshape(-1,6)
t_train = matfile['ttrain2']
X_test = matfile['Xtest2'].reshape(-1,6)
t_test = matfile['ttest2']


matfile = loadmat('matlab_matrix.mat')
A = matfile['A']
print(A)

nState    =   6
nCtrl     =   3

g = 9.81



X_test = X_test[:1000000,:]
t_test = t_test[:1000000,:]

nTest = X_test.shape[0]
Vdot = np.zeros(nTest)

print('nTest: {}'.format(nTest))



print('Predicted!')

for i in range(nTest):

    if i % 10000 == 0:
        print('step {} of {}'.format(i,nTest))

    xdot = np.array([X_test[i,3], X_test[i,4], X_test[i,5], t_test[i,0], t_test[i,1], t_test[i,2]-g])

    Vdot[i] = X_test[i,:].T @ A.T @ A @ xdot

negative_idx = np.argwhere(Vdot <= 0)
positive_idx = np.argwhere(Vdot > 0)


Vdot_negative = Vdot[negative_idx]
Vdot_positive = Vdot[positive_idx]


print('# Positive Vdots: {}'.format(positive_idx.shape[0]))
print('% Positive Vdots: {} %'.format(100*positive_idx.shape[0]/nTest))
print('# Negative Vdots: {}'.format(negative_idx.shape[0]))
print('% Negative Vdots: {} %'.format(100*negative_idx.shape[0]/nTest))
print("Max of Negative Vdots: {}".format(Vdot_negative.max()))


# max_Vdot_idx = np.argmax(Vdot)
# X_vdot_Max = X_test[max_Vdot_idx]
# y_Vdot_Max = y_predicted[max_Vdot_idx]
# y_pred_Vdot_Max = policy.predict(X_vdot_Max.reshape(1,-1))
# print('Vdot.max(): {}'.format(Vdot.max()))
# print('X_vdot_Max: {}'.format(X_vdot_Max))
# print('y_Vdot_Max: {}'.format(y_Vdot_Max))
# print('y_pred_Vdot_Max: {}'.format(y_pred_Vdot_Max))

# Vdot_max_x  = p_x*X_vdot_Max[0]*X_vdot_Max[3]
# Vdot_max_y  = p_y*X_vdot_Max[1]*X_vdot_Max[4]
# Vdot_max_z  = p_z*X_vdot_Max[2]*X_vdot_Max[5]
# Vdot_max_vx = p_vx*X_vdot_Max[3]*y_Vdot_Max[0]
# Vdot_max_vy = p_vy*X_vdot_Max[4]*y_Vdot_Max[1]
# Vdot_max_vz = p_vz*X_vdot_Max[5]*(y_Vdot_Max[2] - g)
# V_dot_max_sum = Vdot_max_x + Vdot_max_y + Vdot_max_z + Vdot_max_vx + Vdot_max_vy + Vdot_max_vz
# print('\n\n')
# print('Vdot_max_x: {}'.format(Vdot_max_x))
# print('Vdot_max_y: {}'.format(Vdot_max_y))
# print('Vdot_max_z: {}'.format(Vdot_max_z))
# print('Vdot_max_vx: {}'.format(Vdot_max_vx))
# print('Vdot_max_vy: {}'.format(Vdot_max_vy))
# print('Vdot_max_vz: {}'.format(Vdot_max_vz))
# print('V_dot_max_sum: {}'.format(V_dot_max_sum))
# print('\n\n')


if positive_idx.shape[0] != 0:
    print('Positive Vdot X_test: {}'.format(X_test[positive_idx[0],:]))

plt.ioff()

plt.figure(1)
plt.plot(Vdot_negative,'.')
plt.plot(Vdot_positive,'.')
plt.legend(['Vdot<=0', 'Vdot>0'], loc='best')
plt.title('Vdot')
plt.xlabel('Index [-]')
plt.ylabel('Vdot [-]')
plt.savefig('{}Vdot.png'.format(saveflag))



plt.figure(2)
ax = plt.axes(projection='3d')
ax.scatter3D(X_test[negative_idx,0], X_test[negative_idx,1], X_test[negative_idx,2])
ax.scatter3D(X_test[positive_idx,0],X_test[positive_idx,1],X_test[positive_idx,2])
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.legend(['Vdot<=0', 'Vdot>0'], loc='best')
plt.savefig('{}_pos.png'.format(saveflag))

plt.figure(3)
plt.plot(X_test[negative_idx,0],X_test[negative_idx,3],'.')
plt.plot(X_test[positive_idx,0],X_test[positive_idx,3],'.')
# plt.scatter(X_test[:,0],X_test[:,3],s=5,c=Vdot)
# plt.colorbar()
plt.legend(['Vdot<=0', 'Vdot>0'], loc='best')
plt.title('Phasespace X')
plt.xlabel('x [m]')
plt.ylabel('vx [m/s]')
plt.savefig('{}_phasespace_x.png'.format(saveflag))

plt.figure(4)
plt.plot(X_test[negative_idx,1],X_test[negative_idx,4],'.')
plt.plot(X_test[positive_idx,1],X_test[positive_idx,4],'.')
plt.legend(['Vdot<=0', 'Vdot>0'], loc='best')
plt.title('Phasespace Y')
plt.xlabel('y [y]')
plt.ylabel('vy [m/s]')
plt.savefig('{}_phasespace_y.png'.format(saveflag))

plt.figure(5)
plt.plot(X_test[negative_idx,2],X_test[negative_idx,5],'.')
plt.plot(X_test[positive_idx,2],X_test[positive_idx,5],'.')
plt.legend(['Vdot<=0', 'Vdot>0'], loc='best')
plt.title('Phasespace Z')
plt.xlabel('z [m]')
plt.ylabel('vz [m/s]')
plt.savefig('{}_phasespace_z.png'.format(saveflag))



# plt.figure(6)
# ax1 = plt.axes(projection='3d')
# ax1.scatter3D(X_test[negative_idx,0],X_test[negative_idx,3],Vdot[negative_idx],s=5)
# ax1.scatter3D(X_test[positive_idx,0],X_test[positive_idx,3],Vdot[positive_idx],c=Vdot[positive_idx],s=5)
# # plt.colorbar()
# # ax.legend(['Vdot<=0', 'Vdot>0'], loc='best')
# # ax.title('Phasespace X')
# ax1.set_xlabel('x [m]')
# ax1.set_ylabel('vx [m/s]')
# ax1.set_zlabel('Vdot [-]')
# plt.savefig('{}_phasespace_x_3d.png'.format(saveflag))
