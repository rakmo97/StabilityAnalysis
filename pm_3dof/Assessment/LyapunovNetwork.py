
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dot, Layer
from tensorflow.keras import Model, Input, models
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt


class LyapunovDense(Layer):
    def __init__(self, num_outputs, epsilon=0.001, name=None):
        super().__init__(name=name)
        self.num_outputs = num_outputs
        self.epsilon = epsilon
        self.G2exist = False

            
    def build(self, input_shape):

        self.ql = int(np.ceil((int(input_shape[-1]) + 1)/2))
        self.G1 = self.add_weight("G1", shape=[self.ql, int(input_shape[-1])])
        
        if self.num_outputs > int(input_shape[-1]):
            self.G2 = self.add_weight("G2", shape=[(self.num_outputs-int(input_shape[-1])), int(input_shape[-1])])
            self.G2exist = True

        elif self.num_outputs < int(input_shape[-1]):
            print('WARNING: out_features > in_features')
        
    def call(self, x):
       
        top_part = tf.transpose(self.G1)@self.G1 + self.epsilon*tf.eye(self.G1.shape[1])

        if self.G2exist:

            W = tf.concat([top_part,self.G2],axis=0)
            y = tf.matmul(x,tf.transpose(W))

        else:
            y = tf.matmul(x,tf.transpose(top_part))

        return tf.nn.tanh(y)   


class LyapunovNetwork(Model):
    def __init__(self):
        super(LyapunovNetwork, self).__init__()

        self.ld1 = LyapunovDense(64)
        self.ld2 = LyapunovDense(64)
        self.ld3 = LyapunovDense(64)

    def call(self, x):
        
        x = self.ld1(x)
        x = self.ld2(x)
        x = self.ld3(x)

        return Dot(axes=1)([x, x])



if __name__ == '__main__':
    
    plt.close('all')
    base_data_folder = 'E:/Research_Data/StabilityAnalysis/'
    # base_data_folder = '/orange/rcstudents/omkarmulekar/StabilityAnalysis/'
    formulation = 'pm_3dof/'
    saveout_filename = base_data_folder + formulation + "NetworkTraining/MinimizedLyapunovNetwork.index"

    # Create an instance of the model
    V_theta = LyapunovNetwork()
    V_theta.load_weights('testsave')
    
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

    
    y_pred = policy.predict(X_test)
    y_pred[:,0] = np.clip(y_pred[:,0], -20, 20)
    y_pred[:,1] = np.clip(y_pred[:,1], -20, 20)
    y_pred[:,2] = np.clip(y_pred[:,2],   0, 20)
    y_pred = tf.cast(y_pred, dtype=tf.float64)

    print('before V_theta prediction')
    V_theta_evaluated = V_theta.predict(X_test)
    print('after V_theta prediction')
    
    V_SOS = X_test[:,0]*X_test[:,0] + X_test[:,1]*X_test[:,1] + X_test[:,2]*X_test[:,2] + X_test[:,3]*X_test[:,3] + X_test[:,4]*X_test[:,4] + X_test[:,5]*X_test[:,5]
    
    
    plt.figure(1)
    plt.subplot(321)
    plt.plot(X_test[:,0],V_theta_evaluated,'.')
    plt.xlabel('x [m]')
    plt.ylabel('V_theta [-]')
    plt.subplot(323)
    plt.plot(X_test[:,1],V_theta_evaluated,'.')
    plt.xlabel('y [m]')
    plt.ylabel('V_theta [-]')
    plt.subplot(325)
    plt.plot(X_test[:,2],V_theta_evaluated,'.')
    plt.xlabel('z [m]')
    plt.ylabel('V_theta [-]')
    plt.subplot(322)
    plt.plot(X_test[:,3],V_theta_evaluated,'.')
    plt.xlabel('vx [m/s]')
    plt.ylabel('V_theta [-]')
    plt.subplot(324)
    plt.plot(X_test[:,4],V_theta_evaluated,'.')
    plt.xlabel('vy [m/s]')
    plt.ylabel('V_theta [-]')
    plt.subplot(326)
    plt.plot(X_test[:,5],V_theta_evaluated,'.')
    plt.xlabel('vz [m/s]')
    plt.ylabel('V_theta [-]')
    plt.suptitle('V_theta evaluations')
    plt.tight_layout()


    plt.figure(2)
    plt.subplot(321)
    plt.plot(X_test[:,0],V_SOS,'.')
    plt.xlabel('x [m]')
    plt.ylabel('V_theta [-]')
    plt.subplot(323)
    plt.plot(X_test[:,1],V_SOS,'.')
    plt.xlabel('y [m]')
    plt.ylabel('V_theta [-]')
    plt.subplot(325)
    plt.plot(X_test[:,2],V_SOS,'.')
    plt.xlabel('z [m]')
    plt.ylabel('V_theta [-]')
    plt.subplot(322)
    plt.plot(X_test[:,3],V_SOS,'.')
    plt.xlabel('vx [m/s]')
    plt.ylabel('V_theta [-]')
    plt.subplot(324)
    plt.plot(X_test[:,4],V_SOS,'.')
    plt.xlabel('vy [m/s]')
    plt.ylabel('V_theta [-]')
    plt.subplot(326)
    plt.plot(X_test[:,5],V_SOS,'.')
    plt.xlabel('vz [m/s]')
    plt.ylabel('V_theta [-]')
    plt.suptitle('V_SOS evaluations')
    plt.tight_layout()





    # Check if Vdot is negative?
    X_test_tensor = tf.constant(X_test)
    
    with tf.GradientTape() as tape:
        tape.watch(X_test_tensor)
        V_pred = V_theta(X_test_tensor)
        
    
    dVdx = tape.batch_jacobian(V_pred, X_test_tensor)
    
    dVdx = tf.reshape(dVdx, [-1,6])
    
    
    nTest = X_test.shape[0]
    Vdot = np.zeros(nTest)
    print('nTest: {}'.format(nTest))
    
    for i in range(nTest):

        if i % 1000 == 0:
            print('step {} of {}'.format(i,nTest))
        
        Vdot[i] = dVdx[i,0]*X_test[i,3] + dVdx[i,1]*X_test[i,4] + dVdx[i,2]*X_test[i,5] + dVdx[i,3]*y_pred[i,0] + dVdx[i,4]*y_pred[i,1] + dVdx[i,5]*(y_pred[i,2] - g)

    

    plt.figure(3)
    plt.plot(Vdot,'.')
    plt.xlabel('Index [-]')
    plt.ylabel('Vdot [-]')
    
    negative_idx = np.argwhere(Vdot <= 0)
    positive_idx = np.argwhere(Vdot > 0)
    
    
    Vdot_negative = Vdot[negative_idx]
    Vdot_positive = Vdot[positive_idx]


    print('# Positive Vdots: {}'.format(positive_idx.shape[0]))
    print('% Positive Vdots: {} %'.format(100*positive_idx.shape[0]/nTest))
    print('# Negative Vdots: {}'.format(negative_idx.shape[0]))
    print('% Negative Vdots: {} %'.format(100*negative_idx.shape[0]/nTest))
    print("Max of Negative Vdots: {}".format(Vdot_negative.max()))
    
        
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
    
    
    
    plt.figure(3)
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_test[:,0],X_test[:,3],Vdot,c=Vdot,s=5)
    # plt.colorbar()
    # ax.legend(['Vdot<=0', 'Vdot>0'], loc='best')
    # ax.title('Phasespace X')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('vx [m/s]')
    ax.set_zlabel('Vdot [-]')
    plt.savefig('{}_phasespace_x_3d.png'.format(saveflag))
        
    
    # Check radially unboundedness?