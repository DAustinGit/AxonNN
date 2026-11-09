
import pandas as pd
data_path = "curated-solubility-dataset"

df = pd.read_csv(data_path+".csv")
df = df.drop(columns=["ID","Name","InChI","InChIKey", "Ocurrences","Group", "SMILES"])
df["H_Bond_density"] = (df["NumHAcceptors"]+df["NumHDonors"])/df["HeavyAtomCount"]
df["polarity_ratio"] = df["TPSA"]/df["MolWt"]
#df = df.drop(columns=["Compound_ID","measured log(solubility:mol/L)","SMILES","userid"])
corr_df = df.corr()
print(corr_df)
#corr_df.to_csv("correlation//"+data_path+"spearman"+".csv")