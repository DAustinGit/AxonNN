import pandas as pd
import cupy as cp
import Axon as ax
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
import time
test_prediction = []

#*******************model peramerters***************************************
data_path = "curated-solubility-dataset"
learning_rate_schedualing = "decay"
Bvals = 0.2
biases_init = "fill" 
weights_init = "He"  
input_features=13
output_features=1 
learning_rate=1e-2
regularisation=None
batch_size=32
lambda_param = 0.001
epochs = 30
file_path = "model_analytics.csv"
dict = [{"activation_function":"GELU", "hidden_features":32},
        {"activation_function":"GELU", "hidden_features":32},
        {"activation_function":"GELU", "hidden_features":32},
        {"activation_function":"GELU", "hidden_features":32},
        {"activation_function":"GELU", "hidden_features":32},
        {"activation_function":"linear", "hidden_features":1}
        ]
#*************************************************************


model_no = 26
params=pd.DataFrame()

data = [{
    
    "learning rate schedualing": learning_rate_schedualing,
    "Bvals": Bvals,
    "weights init": weights_init,
    "biases init": biases_init,
    "input features": input_features,
    "learning rate": learning_rate,
    "regularisation": regularisation,
    "batch_size": batch_size,
    "lambda": lambda_param,
    "model no": model_no,
    "epochs": epochs,
    "data": 'mol_properties.csv',
    "R2 train":0,
    "R2 test":0,
    "MSE test":0,
    "MSE train":0,
    "delta MSE":0,
    "delta R2":0,
    "data":data_path 


}]
params = pd.DataFrame(data)
X_train = pd.read_csv("data//"+data_path+"_x_train.csv")
Y_train = pd.read_csv("data//"+data_path+"_y_train.csv")
X_test = pd.read_csv("data//"+data_path+"_x_test.csv")
Y_test = pd.read_csv("data//"+data_path+"_y_test.csv")
print (X_train.head())


def get_scale_params(train_array):
    """Calculates mean and std from training data only."""
    train_array = np.array(train_array)
    mean = np.mean(train_array, axis=0, keepdims=True)
    std = np.std(train_array, axis=0, keepdims=True, ddof=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std

def apply_scale(array, mean, std):
    """Applies pre-calculated training mean and std to any dataset."""
    return (np.array(array) - mean) / std

target_cols = [
    "NumRotatableBonds",
    "MolWt", "MolLogP", "MolMR", "HeavyAtomCount", "NumHeteroatoms", "NumValenceElectrons",
    "NumAromaticRings", "NumAliphaticRings", "RingCount", "H_Bond_density", "polarity_ratio"
]
X_train_numeric = X_train[target_cols].to_numpy()
X_mean, X_std = get_scale_params(X_train_numeric)
Y_mean, Y_std = get_scale_params(Y_train)
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[target_cols] = apply_scale(X_train[target_cols], X_mean, X_std)
Y_train_scaled= apply_scale(Y_train, Y_mean, Y_std)
X_test_scaled[target_cols] = apply_scale(X_test[target_cols], X_mean, X_std)
Y_test_scaled= apply_scale(Y_test, Y_mean, Y_std)
pipeline = ax.model(learning_rate_schedualing = learning_rate_schedualing, 
                    Bvals = Bvals, biases_init = biases_init,
                    weights_init = weights_init,
                    model = dict, 
                    input_features=input_features, 
                    output_features=1, learning_rate=learning_rate, 
                    regularisation=regularisation, 
                    batch_size=batch_size, 
                    lambda_param=lambda_param)        
print("training model...")
Y_train_scaled = Y_train_scaled.reshape(-1, 1)
Y_test_scaled = Y_test_scaled.reshape(-1, 1)
s = time.process_time()
pipeline.train(x=X_train_scaled, y=Y_train_scaled, epochs=epochs)
p = time.process_time()
print (f"{p-s} seconds")
rows_test,_=X_test_scaled.shape
rows_train,_=X_train_scaled.shape
def r2(predY, expY):
    predY = np.array(predY).ravel()
    expY = np.array(expY).ravel()
    sum_of_squares = np.sum((expY - np.mean(expY))**2)
    sum_of_squared_residuals = np.sum((expY - predY)**2)
    R2 = 1 - (sum_of_squared_residuals / sum_of_squares)
    return R2
def data_MSE (predY, expY):
    predY = np.array(predY).ravel()
    expY = np.array(expY).ravel()
    n= expY.size
    dy = expY-predY
    return (np.sum(dy**2))/n
print("testing model...")
for i in range (rows_test):
  test_pred = pipeline.predict (X_test_scaled[i:i+1])
  #print (f" predicted test y: {test_pred} label: {Y_test[i]}")
  test_prediction.append(float(test_pred[0,0]))
rows,_=X_train_scaled.shape
train_prediction = []
for i in range (rows_train):
  train_pred = pipeline.predict (X_train_scaled[i:i+1])
  #print (f" predicted train y: {train_pred} label: {Y_train[i]}")
  train_prediction.append(float(train_pred[0,0]))
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 20))
MSE = pipeline.MSE()
total_weights = pipeline.total_weights()
analytics = pipeline.model_analytics()
#print (analytics)
DMSE = []
for i in range (1,len(MSE)):
  dMSE2 = (MSE[i]-MSE[-1])**2
  DMSE.append(dMSE2)
axes[0].plot(MSE, marker='o', color='purple')
axes[0].set_title("Mean Squared Error by Epoch")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("MSE")
axes[0].grid(True)
x_cpu = Y_test_scaled.ravel()
y_cpu = np.array(test_prediction).ravel()
m, c = np.polyfit(x_cpu, y_cpu, 1)
axes[1].scatter(Y_test_scaled, test_prediction, marker='o', color='purple')
axes[1].plot(x_cpu, m*x_cpu + c, color='red', linewidth=2)
axes[1].set_title("test set Experimental logS vs Predicted logS")
axes[1].set_xlabel("Experimental LogS")
axes[1].set_ylabel("Predicted logS")
axes[1].grid(True)
x_cpu = Y_train_scaled.ravel()
y_cpu = np.array(train_prediction).ravel()
m, c = np.polyfit(x_cpu, y_cpu, 1)
axes[2].scatter(Y_train_scaled, train_prediction, marker='o', color='purple')
axes[2].plot(x_cpu, m*x_cpu + c, color='red', linewidth=2)
axes[2].set_title("train set Experimental logS vs Predicted logS")
axes[2].set_xlabel("Experimental LogS")
axes[2].set_ylabel("Predicted logS")
axes[2].grid(True)
R2_test = r2(test_prediction, Y_test_scaled)
R2_train = r2(train_prediction, Y_train_scaled)
MSE_test = data_MSE(test_prediction, Y_test_scaled)
MSE_train = data_MSE(train_prediction, Y_train_scaled)
params["R2 test"]=R2_test
params["R2 train"]=R2_train
params["MSE test"]=MSE_test
params["MSE train"]=MSE_train
params["delta R2"]=MSE_train-MSE_test
params["delta MSE"]=MSE_train-MSE_test
print(f"R2 train = {R2_train}")
print(f"R2 test = {R2_test}")
print(f"MSE train = {MSE_train}")
print(f"MSE test = {MSE_test}")

print(f"MSE difference = {MSE_train-MSE_test}")
print(f"R2 difference = {R2_train-R2_test}")
plt.tight_layout()
axes[-1].remove()
axes[3] = fig.add_subplot(1, 4, 4, projection='3d')

gradient_analytics= analytics["layer_gradients"]

print (gradient_analytics.keys())
for key, z_data in gradient_analytics.items():
  x_indeces = range(len(z_data))  
  axes[3].plot(
        x_indeces[-100:],
        z_data[-100:],
        zs=key,    # Positioned at Y = 0, 2, and 4
        zdir='y',    # The plane is perpendicular to the Y-axis
        label=f'layer {key}'
    )
axes[3].set_xlabel('passes')
axes[3].set_zlabel('gradient')
axes[3].set_ylabel('Layer')

#axes[2].plot(gradient_analytics[7][-100:], marker='o', color='purple')
dir = Path(f"Axon_model_analysis/model{model_no}/{data_path}")
file = dir/f"{epochs}epochs_{regularisation}_lambda_param_{learning_rate_schedualing}.png"          
dir.mkdir(parents=True, exist_ok=True)
plt.savefig(file)
plt.show()
with open(f"models_json//model{model_no}.txt", "w") as f:
    json.dump(dict, f, indent=4)
file_exists = os.path.isfile(file_path)
params.to_csv(f"models_json//model_analytics.csv",mode='a',index=False, header=not file_exists)
#pipeline.dump()
