import pandas as pd
from sklearn.model_selection import train_test_split
import cupy as np
import Axon as ax
import matplotlib.pyplot as plt


test_prediction = []

df = pd.read_csv('mol_properties.csv')

Y = df['measured log(solubility:mol/L)']   # Y is feature used for training
X = df.drop(['Compound_ID','measured log(solubility:mol/L)','SMILES'], axis=1)   #X is the feature we are predicting drop logS and axis = 1 --> work in coluns




X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size= 0.20, random_state = 100) #splits the data for train and test sets testing set is 20% of data
#print (X_train)
x_train = X_train.to_numpy()
x_test = X_test.to_numpy()
Y_train = Y_train.to_numpy()
Y_label_test =Y_test
Y_label_train = Y_train
Y_test = Y_test.to_numpy()
y_max = np.max(Y_train)
x_mean = np.mean(x_train, axis=0, keepdims=True)

y_max_test = np.max(Y_test)
x_mean_test = np.mean(x_test, axis=0, keepdims=True)
x_std_test = np.std(x_test, axis=0, keepdims=True)

x_std = np.std(x_train, axis=0, keepdims=True)
x_train_scaled = (x_train-x_mean)/x_std
Y_train_scaled = Y_train/y_max
x_test_scaled = (x_test-x_mean_test)/x_std_test
Y_test_scaled = Y_test/y_max_test




dict = [{"activation_function":"RelU", "hidden_features":48},
        {"activation_function":"RelU", "hidden_features":48},
        {"activation_function":"RelU", "hidden_features":48},
        {"activation_function":"RelU", "hidden_features":48},
        {"activation_function":"RelU", "hidden_features":48},
        {"activation_function":"RelU", "hidden_features":48},
        {"activation_function":"RelU", "hidden_features":48},
        {"activation_function":"linear", "hidden_features":48},


        ]


pipeline = ax.model(model = dict, input_features=8, output_features=1, learning_rate=0.0001, regularisation=None, batch_size=15, lambda_param = 0.01)
                
print("training model...")
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)
pipeline.train(x=x_train_scaled, y=Y_train, epochs=400)


rows_test,_=x_test_scaled.shape
rows_train,_=x_train_scaled.shape
def r2(predY, expY):
    # Convert inputs to numpy arrays to allow vector operations
    predY = np.array(predY)
    expY = np.array(expY)
    
    # 1. Calculate Total Sum of Squares (TSS)
    sum_of_squares = np.sum((expY - np.mean(expY))**2)
    
    # 2. Calculate Residual Sum of Squares (RSS) vectorized (no loop needed)
    sum_of_squared_residuals = np.sum((expY - predY)**2)
    
    # 3. Calculate R²
    R2 = 1 - (sum_of_squared_residuals / sum_of_squares)
    
    return R2



for i in range (rows_test):
  test_pred = pipeline.predict (x_test_scaled[i:i+1])
  #print (f" predicted test y: {test_pred} label: {Y_test[i]}")
  test_prediction.append(float(test_pred[0,0]))


rows,_=x_train_scaled.shape
train_prediction = []
for i in range (rows_train):
  train_pred = pipeline.predict (x_train_scaled[i:i+1])
  #print (f" predicted train y: {train_pred} label: {Y_train[i]}")
  train_prediction.append(float(train_pred[0,0]))


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 6))

MSE = pipeline.MSE()
DMSE = []
for i in range (1,len(MSE)):
  dMSE2 = (MSE[i]-MSE[-1])**2
  DMSE.append(dMSE2)



axes[0].plot(MSE, marker='o', color='purple')

axes[0].set_title("Mean Squared Error by Epoch")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("MSE")
axes[0].grid(True)

axes[1].scatter(Y_label_test, test_prediction, marker='o', color='purple')


axes[1].set_title("test set Experimental logS vs Predicted logS")
axes[1].set_xlabel("Experimental LogS")
axes[1].set_ylabel("Predicted logS")
axes[1].grid(True)
axes[2].scatter(Y_label_train, train_prediction, marker='o', color='purple')
axes[2].set_title("train set Experimental logS vs Predicted logS")
axes[2].set_xlabel("Experimental LogS")
axes[2].set_ylabel("Predicted logS")
axes[2].grid(True)

axes[3].plot(DMSE, marker='o', color='purple')
axes[3].set_title("DMSE")
axes[3].set_xlabel("epochs")
axes[3].set_ylabel("change in MSE")
axes[3].grid(True)

R2_test = r2(test_prediction, Y_label_test)
R2_train = r2(train_prediction, Y_label_train)
print(f"R2 train = {R2_train}")
print(f"R2 test = {R2_test}")
print(f"R2 difference = {R2_train-R2_test}")
plt.tight_layout()
plt.show()

#pipeline.dump()