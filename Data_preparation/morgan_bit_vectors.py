import urllib
import pandas as pd
from sqlalchemy import create_engine, text
from rdkit import Chem
from rdkit.Chem import Lipinski
import streamlit as st
import numpy as np
df = pd.read_csv('AqSolv_data.csv')

def push_data(database: str, table: str, dataframe: pd.DataFrame):
    server = st.secrets['server'] 
    username = st.secrets['username'] 
    password = st.secrets['password']  

    encoded_password = urllib.parse.quote_plus(password)
    connection_url = f"postgresql+psycopg2://{username}:{encoded_password}@{server}:5432/{database}?sslmode=require"
    
    try:
        engine = create_engine(connection_url)
        with engine.connect() as connection:
            print("Successfully connected to Azure postgreSQL SQL Server via SQLAlchemy!")
            df.to_sql(name=table, con=engine, if_exists='replace', index=False)
            return df
            
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return pd.DataFrame()



df['num_NHOH'] = df['SMILES'].apply(lambda x: Lipinski.NHOHCount(Chem.MolFromSmiles(x)) if pd.notnull else None)
df['num_NO'] = df['SMILES'].apply(lambda x: Lipinski.NOCount(Chem.MolFromSmiles(x)) if pd.notnull else None)
df['num_Amide'] = df['SMILES'].apply(lambda x: Lipinski.NumAmideBonds(Chem.MolFromSmiles(x)) if pd.notnull else None)
df['Num_aromatic_heterocycles'] = df['SMILES'].apply(lambda x: Lipinski.NumAromaticHeterocycles(Chem.MolFromSmiles(x)) if pd.notnull else None)
df['Num_aromatic_rings'] = df['SMILES'].apply(lambda x: Lipinski.NumAromaticRings(Chem.MolFromSmiles(x)) if pd.notnull else None)
df['phi'] = df['SMILES'].apply(lambda x: Lipinski.Phi(Chem.MolFromSmiles(x)) if pd.notnull else None)




df = df.drop(['created_at'],axis=1)
print(df.dtypes.value_counts())
#push_data(database='solubility_training', table='mol_properties_exp', dataframe=df)
df.to_csv("molecular_properties_expanded.csv")
#df.to_csv('mol_properties.csv', index=False)