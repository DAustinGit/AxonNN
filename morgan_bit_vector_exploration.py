import pandas as pd
from sklearn.model_selection import train_test_split
import cupy as np
import Axon as ax
import matplotlib.pyplot as plt
import numpy as native_np

test_prediction = []

df = pd.read_csv('AqSolv_morgan_bit_vectors.csv')
#df = pd.read_csv('mol_properties.csv')
df = df.loc[:, (df != 0).any(axis=0)]
#Y = df['solubility_g_L']
df = df.drop(columns=["userid"])
#df.drop(["userid"], axis=1)
Y = df['measured log(solubility:mol/L)']   # Y is feature used for training
X = df.drop(['Compound_ID','measured log(solubility:mol/L)','SMILES'], axis=1)   #X is the feature we are predicting drop logS and axis = 1 --> work in coluns




X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size= 0.20, random_state = 100) #splits the data for train and test sets testing set is 20% of data
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
    "num_Rotatable_Bonds", "num_H_Bond_Acceptors", "num_H_Bond_Doners",
    "num_Nitrogen_Oxygen", "mol_weight", "LogP", "aromatic proportion", "TPSA"
]
# 2. CALCULATE PARAMETERS FROM TRAINING DATA
X_train_numeric = X_train[target_cols].to_numpy()
X_mean, X_std = get_scale_params(X_train_numeric)
Y_mean, Y_std = get_scale_params(Y_train)


X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# 3. APPLY SCALING TO BOTH TRAIN AND TEST datasets
X_train_scaled[target_cols] = apply_scale(X_train[target_cols], X_mean, X_std).get()
Y_train_scaled= apply_scale(Y_train, Y_mean, Y_std)

X_test_scaled[target_cols] = apply_scale(X_test[target_cols], X_mean, X_std).get()
Y_test_scaled= apply_scale(Y_test, Y_mean, Y_std)

print (f"X scaled {X_train_scaled}")
print (f"Y scaled {Y_train_scaled}")
# 4. FIXED PANDAS ASSIGNMENT SYNTAX
# Specific columns must be passed inside an inner list: [['col1', 'col2']]


dict = [{"activation_function":"ReLU", "hidden_features":1120},
        {"activation_function":"ReLU", "hidden_features":1120},
        {"activation_function":"ReLU", "hidden_features":600},
        {"activation_function":"ReLU", "hidden_features":600},
        {"activation_function":"ReLU", "hidden_features":600},
        {"activation_function":"ReLU", "hidden_features":600},
        {"activation_function":"ReLU", "hidden_features":600},
        {"activation_function":"linear", "hidden_features":1}


        ]


pipeline = ax.model(learning_rate_schedualing = True, Bvals = 0.0, biases_init = "fill", weights_init = "He", model = dict, input_features=264, output_features=1, learning_rate=5e-5, regularisation=None, batch_size=32, lambda_param = 0.01)
                
print("training model...")
Y_train_scaled = Y_train_scaled.reshape(-1, 1)
Y_test_scaled = Y_test_scaled.reshape(-1, 1)


pipeline.train(x=X_train_scaled, y=Y_train_scaled, epochs=400)


rows_test,_=X_test_scaled.shape
rows_train,_=X_train_scaled.shape
def r2(predY, expY):
    # Convert inputs to numpy arrays to allow vector operations
    predY = np.array(predY).ravel()
    expY = np.array(expY).ravel()
    
    # 1. Calculate Total Sum of Squares (TSS)
    sum_of_squares = np.sum((expY - np.mean(expY))**2)
    
    # 2. Calculate Residual Sum of Squares (RSS) vectorized (no loop needed)
    sum_of_squared_residuals = np.sum((expY - predY)**2)
    
    # 3. Calculate R²
    R2 = 1 - (sum_of_squared_residuals / sum_of_squares)
    
    return R2



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
print (analytics)
DMSE = []
for i in range (1,len(MSE)):
  dMSE2 = (MSE[i]-MSE[-1])**2
  DMSE.append(dMSE2)


axes[0].plot(MSE, marker='o', color='purple')
axes[0].set_title("Mean Squared Error by Epoch")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("MSE")
axes[0].grid(True)
print (type(Y_test_scaled), type(test_prediction))
x_cpu = Y_test_scaled.get().ravel()
y_cpu = native_np.array(test_prediction).ravel()
m, c = native_np.polyfit(x_cpu, y_cpu, 1)

axes[1].scatter(Y_test_scaled.get(), test_prediction, marker='o', color='purple')
axes[1].plot(x_cpu, m*x_cpu + c, color='red', linewidth=2)

axes[1].set_title("test set Experimental logS vs Predicted logS")
axes[1].set_xlabel("Experimental LogS")
axes[1].set_ylabel("Predicted logS")
axes[1].grid(True)


x_cpu = Y_train_scaled.get().ravel()
y_cpu = native_np.array(train_prediction).ravel()
m, c = native_np.polyfit(x_cpu, y_cpu, 1)


axes[2].scatter(Y_train_scaled.get(), train_prediction, marker='o', color='purple')
axes[2].plot(x_cpu, m*x_cpu + c, color='red', linewidth=2)
axes[2].set_title("train set Experimental logS vs Predicted logS")
axes[2].set_xlabel("Experimental LogS")
axes[2].set_ylabel("Predicted logS")
axes[2].grid(True)
'''''
axes[3].plot(gradients, marker='o', color='purple')
axes[3].set_title("total_weights")
axes[3].set_xlabel("layers")
axes[3].set_ylabel("total_weights")
axes[3].grid(True)
'''
R2_test = r2(test_prediction, Y_test_scaled)
R2_train = r2(train_prediction, Y_train_scaled)
print(f"R2 train = {R2_train}")
print(f"R2 test = {R2_test}")
print(f"R2 difference = {R2_train-R2_test}")
plt.tight_layout()
axes[-1].remove()
axes[3] = fig.add_subplot(1, 4, 4, projection='3d')

gradient_analytics= analytics["layer_gradients"]
print (analytics.keys())
print (gradient_analytics.keys())
for key, z_data in gradient_analytics.items():
  x_indeces = range(len(z_data))  
  axes[3].plot(
        x_indeces,
        z_data,
        
        
        zs=key,    # Positioned at Y = 0, 2, and 4
        zdir='y',    # The plane is perpendicular to the Y-axis
        label=f'layer {key}'
    )
axes[3].set_xlabel('passes')
axes[3].set_zlabel('gradient')
axes[3].set_ylabel('Layer')


plt.show()
#pipeline.dump()
