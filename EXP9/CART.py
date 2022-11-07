import numpy as np
import pandas as pd

df = pd.read_csv("playgolf_data.csv")
df

df.dtypes
df.info()

df_getdummy=pd.get_dummies(data=df, columns=['Temperature', 'Humidity', 'Outlook', 'Wind'])
df_getdummy
    