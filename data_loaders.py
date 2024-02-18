### Data loaders for the irrigation mapping project

# Import the required modules
import netCDF4 as nc
import numpy as np
import torch
from sklearn.model_selection import train_test_split

### Function to load the WRF training data
def load_WRFSCM_training_data(fpath, seed=None):

    # Open the file and extract variables
    fn = nc.Dataset(fpath)
    vegfra = fn.variables['VEGFRA'][:] # 1-D
    irr = fn.variables['IRR'][:]        # 1-D (the target)
    sm = fn.variables['SM'][:]          # 1-D
    lst = fn.variables['LST'][:]        # 2-D (samples, features)
    time = fn.variables['SimTime'][:]   # 2-D (samples, features)
    
    # Normalize the variables
    vegfra = (vegfra-np.mean(vegfra))/np.std(vegfra)
    sm = (sm-np.mean(sm))/np.std(sm)
    lst = (lst-np.mean(lst, axis=0))/np.std(lst, axis=0)
    irr2 = (irr-np.mean(irr))/np.std(irr)
       
    # Stack into a single array
    X = np.stack([vegfra, sm]+list([lst[:,i] for i in range(lst.shape[1])]), axis=1).astype('double')
    y = irr2
    
    # Do the train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=seed)
    
    # Perform the normalization
    Xbar = np.mean(X_train, axis=0)
    Xsig = np.std(X_train, axis=0)
    ybar = np.mean(y_train)
    ysig = np.std(y_train)

    for i in range(X_train.shape[1]):
        X_train[:,i] = (X_train[:,i]-Xbar[i])/Xsig[i]
        X_test[:,i] = (X_test[:,i]-Xbar[i])/Xsig[i]

    y_train = (y_train-ybar)/ysig
    y_test = (y_test-ybar)/ysig

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # Return the training inputs and the targets
    return X_train, X_test, y_train, y_test, ybar, ysig