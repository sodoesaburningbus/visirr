### This script trains the model
### Christopher Phillips

##### START OPTIONS #####

# Location of training data
fpath = 'wrfvars.2018-07-21_11-00-00.nc'

# Epochs and batch size
n_epochs = 50
batch_size = 40

# How many times to repeat training?
n_training = 10

#####  END OPTIONS  #####


### Import modules
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

### Import model files
from model import irr_net
from data_loaders import load_WRFSCM_training_data

### Load the data
X_train, X_test, y_train, y_test, ybar, ysig = load_WRFSCM_training_data(fpath, seed=44)


# Set best loss to inifinity
best_loss = np.inf
    
# Load the network
model = irr_net()


optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
# Loop over epochs and batches
for epoch in range(n_epochs):

    # Set model to training mode
    model.train()

    for i in range(0, len(X_train), batch_size):

        # Extract batch
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        # Make a prediction
        y_pred = model(X_batch)        

        # Compute the loss
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Check if model improved
    model.eval()
    y_pred = model(X_test)
    loss = loss_fn(y_pred, y_test)
    loss = float(loss)

    # Store best weights
    if (loss < best_loss):
        best_loss = loss
        best_weights = copy.deepcopy(model.state_dict())

# Load best model
model.load_state_dict(best_weights)

# Do predictions and print accuracies
y_pred = np.array(np.squeeze(model(X_test).detach().numpy()))
y_test = np.array(np.squeeze(y_test.detach().numpy()))

r2 = np.mean(y_pred*y_test)

y_pred = y_pred*ysig+ybar
y_test = y_test*ysig+ybar

rmse = np.sqrt(np.mean((y_pred-y_test)**2))
mae = np.mean(np.abs(y_pred-y_test))
mbe = np.mean(y_pred-y_test)

print('-----------------------------------')
print('RMSE: ', rmse)
print('MAE: ', mae)
print('MBE: ', mbe)
print('R2', r2)

# Save the best weights
torch.save(model.state_dict(), 'IRR_NN.pt')