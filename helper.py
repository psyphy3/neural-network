import numpy as np
def sig(x):
    y=1.0/(1.0+np.exp(-x))
    return y

def dsig(x):
    y= sig(x)*(1-sig(x))
    return y
    
def relu(x):
    return max(x,0)


    