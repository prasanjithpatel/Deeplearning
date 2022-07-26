import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

I = np.array([[0.2, 0.3, 0.4, 0.5, 0.6]]).T
V = np.array([[1.23, 1.38, 2.06, 2.47, 3.17]]).T
plt.scatter(I, V)
plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.show()

#batch least squares method

H = np.ones((5, 2))
H[:, 0] = I.ravel()
x_ls = inv(H.T.dot(H)).dot(H.T.dot(V))
print('The slope and offset parameters of the best-fit line (i.e., the resistance and offset) are [R, b]:')
print(x_ls[0, 0])
print(x_ls[1, 0])

# Plot line.
I_line = np.arange(0, 0.8, 0.1).reshape(8, 1)
V_line = x_ls[0]*I_line + x_ls[1]

plt.scatter(I, V)
plt.plot(I_line, V_line)
plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.show()


## Recursive Solution

# Initialize the 2x1 parameter vector x (i.e., x_0).
# x_k = ...
x_k=np.array([[4.0,1.0]])


#Initialize the 2x2 covaraince matrix (i.e. P_0). Off-diangonal elements should be zero.
# P_k = ...
P_k = np.array([[4, 10.0], [0, 0.2]])

# Our voltage measurement variance (denoted by R, don't confuse with resistance).
R_k = np.array([[0.0225]])

# Pre allocate space to save our estimates at every step.
num_meas = I.shape[0]
x_hist = np.zeros((num_meas + 1, 2))
p_hist = np.zeros((num_meas + 1, 2, 2))

x_hist[0] = x_k
p_hist[0] = P_k

# Iterate over all the available measurements.
for k in range(num_meas):
    # Construct H_k (Jacobian).
    # H_k = ...
    H_k = np.array([[I[k], 1.0]],dtype='float')
    
    #sub parts for gain 
    a=np.dot(H_k,p_hist[k-1])
    #print(a.shape,"a_shape")
    b=np.dot(a,H_k.transpose())
    #print(b.shape,"b_shape")
    b=b+R_k
    b=inv(b)
    #print("final b_sphae",b.shape)
    c=np.dot(p_hist[k-1],H_k.transpose())
    #print("before bracket",c.shape)
    k_k=np.dot(c,b)

    #K_k = np.dot(p_hist[k-1] , np.dot(H_k.transpose() , inv(np.dot(H_k,np.dot(p_hist[k-1],H_k.transpose())) + R_k)))

                    
    # Update our estimate.
    # x_k = ...
    x_k = x_hist[k-1] + np.dot(k_k,(V[k]-np.dot(H_k,x_hist[k-1])))
 
    # Update our uncertainty (covariance)
    # P_k = ...   
    p_k=np.dot((np.eye(2,2)-np.dot(k_k,H_k)),p_hist[k-1])

    # Keep track of our history.
    p_hist[k + 1] = P_k
    x_hist[k + 1] = x_k
    
print('The slope and offset parameters of the best-fit line (i.e., the resistance and offset) are [R, b]:')
print(x_k[0, 0])
print(x_k[1, 0])
plt.scatter(I, V, label='Data')
plt.plot(I_line, V_line, label='Batch Solution')
plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.grid(True)

I_line = np.arange(0, 0.8, 0.1).reshape(8, 1)

for k in range(num_meas):
    V_line = x_hist[k, 0]*I_line + x_hist[k, 1]
    plt.plot(I_line, V_line, label='Measurement {}'.format(k))

plt.legend()
plt.show()
