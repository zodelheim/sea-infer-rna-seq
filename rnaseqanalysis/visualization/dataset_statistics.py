import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler


fdir_raw = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/raw/")
fdir_processed = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/interim")
fdir_traintest = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/processed") / 'sex'
fdir_external = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/external")


use_CV = True

model_type = 'catboost'
model_type = 'xgboost'

sex = 'chrXY'
sex = 'autosome'


fname = Path("heart.merged.TPM.preprocessed.csv")
data = pd.read_csv(fdir_external / 'HEART' / 'reg' / fname, index_col=0)
data_header = pd.read_csv(fdir_external / 'HEART' / 'reg' / 'SraRunTable.txt', sep=',')

# data = pd.read_csv(fdir_traintest / f'geuvadis.preprocessed.{sex}.csv', index_col=0)
# data_header = pd.read_csv(fdir_raw / 'Geuvadis.SraRunTable.txt', index_col=0)


features = pd.read_csv(fdir_processed / f'feature_importance.{model_type}.{sex}.csv', index_col=0)
features = features.loc[features.index.intersection(data.columns)]
n_features = 100
# print(features.iloc[:n_features].index)
data = data[features.iloc[:n_features].index]

# index_to_drop = data.index[((((data - data.mean()) / data.std()).abs() > 6).sum(axis=1) > 0).values]
# data = data.drop(index_to_drop)

# index_to_drop = data.index[((data < -12).sum(axis=1) > 0).values]
# data = data.drop(index_to_drop)
data[data < -12] = pd.NA

# data_values = StandardScaler().fit_transform(data)
# data = pd.DataFrame(data=data_values, columns=data.columns, index=data.index)


print(data)

sns.violinplot(data)
sns.stripplot(data, color='black')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
