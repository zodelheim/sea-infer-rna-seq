import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# import scanpy as sc

# sc.set_figure_params(dpi=80, color_map='viridis')
# sc.settings.verbosity = 2
# sc.logging.print_versions()


fdir_raw = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/raw/")
fdir_processed = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/interim")
fdir_traintest = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/processed") / 'sex'
fdir_external = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/external")

sex = 'chrXY'
# sex = 'chrX'
# sex = 'chrY'
# sex = 'autosome'

data_raw = pd.read_csv(fdir_raw / 'Geuvadis.all.csv', index_col=0).T
data_raw = data_raw.astype(np.float32)

# data = pd.read_csv(fdir_traintest / f'geuvadis.preprocessed.{sex}.csv', index_col=0)

data = data_raw
# data = data + 1e-6
# data = np.log(data)

print(f"{data.shape=}")

data_header = pd.read_csv(fdir_raw / 'Geuvadis.SraRunTable.txt', index_col=0)
data_header = data_header[[
    'Sex',
]]
data_header = data_header.loc[data.index]
data['Sex'] = data_header['Sex']

features = pd.read_csv(fdir_processed / f'feature_importance.{"xgboost"}.{sex}.csv', index_col=0).index
print(f"{features.shape=}")

transcripts_x = pd.read_csv(fdir_processed / "all_transcripts.chrX.csv", index_col=0).values.ravel()
transcripts_y = pd.read_csv(fdir_processed / "all_transcripts.chrY.csv", index_col=0).values.ravel()

print(f"{transcripts_x.shape=}")
print(f"{transcripts_y.shape=}")

intersection_x = data.columns.intersection(transcripts_x)
intersection_y = data.columns.intersection(transcripts_y)

print(f"{intersection_x.shape=}")
print(f"{intersection_y.shape=}")

intersection_x_top = intersection_x.intersection(features)
intersection_y_top = intersection_y.intersection(features)

print(f"{intersection_x_top.shape=}")
print(f"{intersection_y_top.shape=}")

# print(data[intersection_y].mean())
# # print(data[intersection_y.union(["Sex"])].melt('Sex'))
# sns.histplot(data[intersection_y].mean().values)
# # plt.xticks(rotation=90)
# plt.show()

# diff = data.loc[data['Sex'] == 'M'].drop(columns="Sex").mean() - data.loc[data['Sex'] == 'F'].drop(columns="Sex").mean()
# sns.histplot(data=diff)
# plt.show()
# # print(diff)
# exit()


data_melted = data[intersection_y.union(["Sex"])].melt("Sex")
data_melted.index = data_melted['variable']
# data_melted['value'] += 1e-6


# data_melted['value'] = np.log(data_melted['value'])
# data_melted = data_melted.set_index('variable')
# print(data_melted.loc[intersection_x[:10]])
# exit()

# fig, axs = plt.subplots()

# sns.stripplot(
#     data=data_melted.loc[intersection_x[:10]],
#     # data=data[["ENST00000711172.1"] + ['Sex']].melt('Sex'),
#     x='variable', y='value', hue='Sex',
#     log_scale=True, ax=axs
# )
print(intersection_y)
g = sns.violinplot(
    # data=data_melted.loc[intersection_y[:20]],
    # data=data_melted,
    data=data[['ENST00000711179.1', 'ENST00000711233.1', 'MSTRG.36698.1',
               'MSTRG.36700.1', 'ENST00000711204.1', 'ENST00000711172.1'] + ['Sex']].melt('Sex'),
    x='variable', y='value', hue='Sex',
    split=True, inner="quart", bw_adjust=.5,
    log_scale=False
)

# g = sns.catplot(
#     data=data_melted.loc[intersection_y[:20]],
#     x='variable', y='value', hue='Sex',
#     # split=True, inner="quart", bw_adjust=.5,
#     kind='violin', split=True, inner='quart', bw_adjust=0.5,
#     log_scale=True
# )
# g.figure.set_size_inches(16, 4)

sns.move_legend(g, 'center left', bbox_to_anchor=(1, 0.5))
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
