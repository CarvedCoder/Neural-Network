import numpy as np
import matplotlib.pyplot as plt

def generate_spiral_data(points,classes):
    X = np.zeros((points * classes,2))
    y = np.zeros(points*classes)
    for j in range(classes):
        ix = range(points * j,points * (j+1))
        r = np.linspace(0.0,1,points)
        t = np.linspace(j*4,(j+1)*4,points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r*np.sin(t*2.5),r*np.cos(t*2.5)]
        y[ix] = j
    return X,y

X,y = generate_spiral_data(250,2)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()
np.savetxt("Spiral_X.csv",X,delimiter=",")
np.savetxt("Spiral_y.csv",y.reshape(-1,1),delimiter=",")


