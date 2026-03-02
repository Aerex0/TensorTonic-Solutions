import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def _yPred(w,X,b):
    z = np.dot(X,w)+b
    return _sigmoid(z)

def _BCE(y_true,y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred,eps,1-eps)
    loss = -(y_true*np.log(y_pred)+(1-y_true)*(np.log(1-y_pred)))
    return np.mean(loss)

def _grads(X,y_pred,y_true):
    m = X.shape[0]
    dw = (1/m)*np.dot(X.T,y_pred-y_true)
    db = (1/m)*np.sum(y_pred-y_true)
    return dw,db

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    w = np.zeros(X.shape[1])
    b = 0.0

    for epoch in range(steps):
        y_pred = _yPred(w,X,b)
        loss = _BCE(y,y_pred)

        dL_dw,dL_db= _grads(X,y_pred,y)

        w-=lr*dL_dw
        b-=lr*dL_db

    return w,b

        
    