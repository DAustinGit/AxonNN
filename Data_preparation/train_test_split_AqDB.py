import pandas as pd
from sklearn.model_selection import train_test_split
data_path = "curated-solubility-dataset"
df = pd.read_csv("data//"+data_path+".csv")
df["H_Bond_density"] = (df["NumHAcceptors"]+df["NumHDonors"])/df["HeavyAtomCount"]
df["polarity_ratio"] = df["TPSA"]/df["MolWt"]
df = df.drop(columns=["ID","Name","InChI","InChIKey", "Ocurrences","Group", "SMILES",  "BertzCT", "LabuteASA", "BalabanJ","NumHAcceptors","NumHDonors", "TPSA", "BalabanJ", "SD"])
Y = df['Solubility'] 
X = df.drop(['Solubility'], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size= 0.20, random_state = 100) #splits the data for train and test sets testing set is 20% of data
print (X_train.head())
X_train.to_csv("data//"+data_path+"_x_train.csv",index=False)
Y_train.to_csv("data//"+data_path+"_y_train.csv",index=False)
X_test.to_csv("data//"+data_path+"_x_test.csv",index=False)
Y_test.to_csv("data//"+data_path+"_y_test.csv",index=False)