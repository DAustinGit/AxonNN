import pandas as pd
import cupy as cp
import Axon as ax
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
import time
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, average_precision_score
test_prediction = []

#*******************model peramerters***************************************
threshold = 0.4
data_path = "Bankrupcy"
learning_rate_schedualing = None
Bvals = 0.0
biases_init = "fill" 
weights_init = "He"  
input_features=93
output_features=1 
learning_rate=5e-3
regularisation=None
batch_size=32
lambda_param = 0.001
loss_type = "BCEwithlogits"
epochs = 1500
file_path = "models/classifier/"
dict = [{"activation_function":"ReLU", "hidden_features":93},
        {"activation_function":"ReLU", "hidden_features":20},
        {"activation_function":"ReLU", "hidden_features":10},
        {"activation_function":"linear", "hidden_features":1}
        ]
#*************************************************************


model_no = 28
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
    "data":data_path, 
    "loss_type":loss_type,
    "threshold":threshold


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


X_train_numeric = X_train.to_numpy()
X_mean, X_std = get_scale_params(X_train_numeric)
Y_mean, Y_std = get_scale_params(Y_train)
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled = apply_scale(X_train, X_mean, X_std)
Y_train_scaled= apply_scale(Y_train, Y_mean, Y_std)
X_test_scaled = apply_scale(X_test, X_mean, X_std)
Y_test_scaled= apply_scale(Y_test, Y_mean, Y_std)
pipeline = ax.model(learning_rate_schedualing = learning_rate_schedualing, 
                    loss_type = loss_type,
                    Bvals = Bvals, biases_init = biases_init,
                    weights_init = weights_init,
                    model = dict, 
                    input_features=input_features, 
                    output_features=1, learning_rate=learning_rate, 
                    regularisation=regularisation, 
                    batch_size=batch_size, 
                    lambda_param=lambda_param)        
print("training model...")

Y_train = Y_train
Y_test = Y_test
s = time.process_time()
Y_train = np.array(Y_train)
pipeline.train(x=X_train_scaled, y=Y_train, epochs=epochs)
p = time.process_time()
print (f"{p-s} seconds")
rows_test,_=X_test_scaled.shape
rows_train,_=X_train_scaled.shape
def validation(predY, trueY):
    df_validation = pd.DataFrame({"truth":trueY.ravel(), "prediction":predY.ravel()})
    p, t = df_validation["prediction"], df_validation["truth"]
    df_validation["true positive"]  = ((p == 1) & (t == 1)).astype(int)
    df_validation["false positive"] = ((p == 1) & (t == 0)).astype(int)
    df_validation["false negative"] = ((p == 0) & (t == 1)).astype(int)
    df_validation["true negative"]  = ((p == 0) & (t == 0)).astype(int)
    print (df_validation)
    df_validation=df_validation[["true positive", "false positive", "false negative", "true negative"]].sum()
    return df_validation
def data_MSE (predY, expY):
    predY = np.array(predY).ravel()
    expY = np.array(expY).ravel()
    n= expY.size
    dy = expY-predY
    return (np.sum(dy**2))/n

Y_test = np.array(Y_test)
print("testing model...")

def prediction(x: np.ndarray, threshold: float, model=pipeline):
    predictions = []
    scores = []
    rows,_ = x.shape
    for i in range (rows):
        preds = model.predict(x[i:i+1], output_activation="sigmoid").get()
        binary = (preds > threshold).astype(int)
        predictions.append(binary)
        scores.append(float(preds[0]))
    #print(scores)
    return np.array(predictions), np.array(scores)
   
test_prediction, test_scores = prediction(x=X_test_scaled, threshold=threshold, model=pipeline)
train_prediction, train_scores = prediction(x=X_train_scaled, threshold=threshold, model=pipeline)
test_total = len(test_prediction)
train_total = len(train_prediction)
test_validation = validation(trueY=Y_test, predY=test_prediction)
train_validation = validation(trueY=Y_train, predY=train_prediction)
test_accurecy = (test_validation["true positive"]+test_validation["true negative"])/test_total
train_accurecy = (train_validation["true positive"]+train_validation["true negative"])/train_total
test_precision = test_validation["true positive"]/(test_validation["true positive"]+test_validation["false positive"])
train_precision = train_validation["true positive"]/(train_validation["true positive"]+train_validation["false positive"])
cm_test = confusion_matrix(Y_test.ravel(), test_prediction.ravel())
cm_train = confusion_matrix(Y_train.ravel(), train_prediction.ravel())

disp1 = ConfusionMatrixDisplay(
    confusion_matrix=cm_test, display_labels=["Not Bankrupt (0)", "Bankrupt (1)"]
)
disp2 = ConfusionMatrixDisplay(
    confusion_matrix=cm_train, display_labels=["Not Bankrupt (0)", "Bankrupt (1)"]
)
report = classification_report(Y_test.ravel(), test_prediction.ravel(), target_names=["not bankrupt","bankrupt"], output_dict=True)

print (type(report))
df_report = pd.DataFrame(report)
print (df_report)
print (f"train: {train_validation}")
print (f"test {test_validation}")
print (f"train: {test_accurecy}%")
print (f"test {train_accurecy}%")
roc_auc = roc_auc_score(Y_test.ravel(),test_scores.ravel())
pr_auc,_,baseline = average_precision_score(Y_test.ravel(),test_prediction.ravel()), " baseline:", Y_test.ravel().mean()
print("ROC_AUC:", roc_auc)
print("PR_AUC: ", pr_auc)
report["ROC_AUC"] = roc_auc
report["PR_AUC"] = pr_auc
report["baseline"] = baseline



fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 20))
training_error = pipeline.training_error()
total_weights = pipeline.total_weights()
analytics = pipeline.model_analytics()
axes[0].plot(training_error, marker='o', color='purple')
axes[0].set_title("Binary Cross Entropy by batch")
axes[0].set_xlabel("batches")
axes[0].set_ylabel("BCE")
axes[0].grid(True)
axes[1].set_title("Test confusion matrix")
axes[2].set_title("Train confusion matrix")

disp1.plot(ax=axes[1], cmap="Blues", values_format="d")
disp2.plot(ax=axes[2], cmap="Greens", values_format="d")
axes[-1].remove()
axes[2] = fig.add_subplot(1, 4, 4, projection='3d')

gradient_analytics= analytics["layer_gradients"]
for key, z_data in gradient_analytics.items():
  x_indeces = range(len(z_data))
  axes[2].plot(
     
        x_indeces[-100:],
        z_data[-100:],
        zs=key,    # Positioned at Y = 0, 2, and 4
        zdir='y',    # The plane is perpendicular to the Y-axis
        label=f'layer {key}'
    )
axes[2].set_xlabel('passes')
axes[2].set_zlabel('gradient')
axes[2].set_ylabel('Layer')

#axes[2].plot(gradient_analytics[7][-100:], marker='o', color='purple'
with open(f"{file_path}//model{model_no}.txt", "w") as f:
    json.dump(dict, f, indent=4)
df = pd.read_csv(file_path+"model_parameters.csv")
last_row_index = df["index"].iloc[-1]
print(last_row_index)
new_index = int(last_row_index)+1

#new_index = 0
params["index"] = new_index
dir = Path(f"{file_path}model{model_no}/{data_path}")
dir.mkdir(parents=True, exist_ok=True)    
file = dir/f"{model_no} {new_index}.png"       
file_exists = os.path.isfile(file_path+"model_parameters.csv")
#file_exists = os.path.isfile(file_path+"model_analytics.csv")
plt.savefig(file)
plt.show()

df_report.to_csv(f"models//classifier//model_analytics_run_{new_index}.csv",mode='a',index=True, header=True,)
params.to_csv(file_path+"model_parameters.csv",mode='a',index=True,  header=not file_exists)

#pipeline.dump()