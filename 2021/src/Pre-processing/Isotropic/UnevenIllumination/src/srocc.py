import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

df_sdv = pd.read_csv(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/res/sdv2.csv', delimiter=',', header=None)

df_niqe = pd.read_csv(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/res/niqes.csv', delimiter='\t', header=None)

df_jnc = pd.read_csv(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/res/jnc_plus.csv', delimiter='\t', header=None)

df_alc = pd.read_csv(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/res/ALC.csv', delimiter='\t', header=None)


df_agic = pd.read_csv(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/res/agic.csv', delimiter='\t', header=None)

df_ihed = pd.read_csv(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/res/ihed_corected_done.csv', delimiter=',', header=None)

df_mosEx = pd.read_csv(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/res/mosEx.csv', delimiter=' ', header=None)

df_mosNonEx = pd.read_csv(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/res/mosNonEx.csv', delimiter=' ', header=None)

# estimated coefficients of different models
df_ihed.columns = ['file', 'value']
df_agic.columns = ['file', 'value']
df_sdv.columns = ['file', 'value']
df_niqe.columns = ['file', 'value']
df_jnc.columns = ['file', 'value']
df_alc.columns = ['file', 'value']

# MOS coefficient of the effect of uneven illumination on the image quality (Expert)
df_mosEx.columns = ['file', 'value']

# Mos coefficient of the effect of uneven illumination on the image quality (Non-Expert)
df_mosNonEx.columns = ['file', 'value']

# sorted values of the coefficient of the effect of uneven illumination on the image quality
sorted_mosEx = df_mosEx.sort_values(by=['file'], ascending=True).values[:, 1]
sorted_mosNonEx = df_mosNonEx.sort_values(
    by=['file'], ascending=True).values[:, 1]

sorted_ihed = df_ihed.sort_values(by=['file'], ascending=True).values[:, 1]
sorted_agic = df_agic.sort_values(by=['file'], ascending=True).values[:, 1]
sorted_sdv = df_sdv.sort_values(by=['file'], ascending=True).values[:, 1]
sorted_niqe = df_niqe.sort_values(by=['file'], ascending=True).values[:, 1]
sorted_jnc = df_jnc.sort_values(by=['file'], ascending=True).values[:, 1]
sorted_alc = df_alc.sort_values(by=['file'], ascending=True).values[:, 1]

sorted_jnc_plus = sorted_jnc*sorted_ihed

# Calcalate the Spearman correlation coefficient
print("SROCC between each measure and MOS expert:")

# print("Spearman correlation coefficient between SDV and MOS expert==>",
#       stats.spearmanr(sorted_sdv, sorted_mosEx))
# print("Spearman correlation coefficient between AGIC and MOS expert==>",
#       stats.spearmanr(1/sorted_agic, sorted_mosEx))
print("Spearman correlation coefficient between IHED and MOS expert==>",
      stats.spearmanr(1/sorted_ihed, sorted_mosEx))
print("Spearman correlation coefficient between ALC and MOS expert==>",
      stats.spearmanr(1/sorted_alc, sorted_mosEx))

# print("Spearman correlation coefficient between NIQE and MOS expert==>",
#       stats.spearmanr(1/sorted_niqe, sorted_mosEx))

print("SROCC between each measure and MOS non-expert:")
# print("Spearman correlation coefficient between SDV and MOS non-expert==>",
#       (stats.spearmanr(sorted_sdv, sorted_mosNonEx)))
# print("Spearman correlation coefficient between AGIC and MOS non-expert==>",
#       (stats.spearmanr(1/sorted_agic, sorted_mosNonEx)))
print("Spearman correlation coefficient between IHED and MOS non-expert==>",
      (stats.spearmanr(1/sorted_ihed, sorted_mosNonEx)))

print("Spearman correlation coefficient between ALC and MOS non-expert==>",
      (stats.spearmanr(1/sorted_alc, sorted_mosNonEx)))      
# print("Spearman correlation coefficient between NIQE and MOS non-expert==>",
#       (stats.spearmanr(1/sorted_niqe, sorted_mosNonEx)))


# Calcalate the Pearson correlation coefficient
print("LCC between each measure and MOS expert:")
# print("Pearson correlation coefficient between SDV and MOS expert==>",
#       (stats.pearsonr(sorted_sdv, sorted_mosEx)))
# print("Pearson correlation coefficient between AGIC and MOS expert==>",
#       (stats.pearsonr(1/sorted_agic, sorted_mosEx)))
print("Pearson correlation coefficient between IHED and MOS expert==>",
      (stats.pearsonr(1/sorted_ihed, sorted_mosEx)))

print("Pearson correlation coefficient between ALC and MOS expert==>",
      (stats.pearsonr(1/sorted_alc, sorted_mosEx)))
# print("Pearson correlation coefficient between NIQE and MOS expert==>",
#       (stats.pearsonr(1/sorted_niqe, sorted_mosEx)))

print("LCC between each measure and MOS non-expert:")
# print("Pearson correlation coefficient between SDV and MOS non-expert==>",
#       (stats.pearsonr(sorted_sdv, sorted_mosNonEx)))
# print("Pearson correlation coefficient between AGIC and MOS non-expert==>",
#       (stats.pearsonr(1/sorted_agic, sorted_mosNonEx)))
print("Pearson correlation coefficient between IHED and MOS non-expert==>",
      (stats.pearsonr(1/sorted_ihed, sorted_mosNonEx)))

print("Pearson correlation coefficient between ALC and MOS non-expert==>",
      (stats.pearsonr(1/sorted_alc, sorted_mosNonEx)))

# print("Pearson correlation coefficient between NIQE and MOS non-expert==>",
#       (stats.pearsonr(1/sorted_niqe, sorted_mosNonEx)))

# x = np.array(sorted_ihed, dtype=float).reshape(-1, 1)
# y = np.array(sorted_mosEx, dtype=float)
# print(np.shape(y))
# sns_plot = sns.regplot(x=x, y=y, logistic=True, scatter_kws={
#                        'color': 'black'}, line_kws={'color': 'red'})

# LogR = LogisticRegression(
#     max_iter=1000, multi_class='multinomial', solver='lbfgs')

# lab_enc = preprocessing.LabelEncoder()
# training_scores_encoded = lab_enc.fit_transform(y)

# LogR.fit(x, training_scores_encoded)
# X_test = np.linspace(-5, 10, 120)

# loss = expit(X_test * LogR.coef_ + LogR.intercept_).ravel()
# plt.plot(X_test, loss, color="red", linewidth=3)

# # matplotlib scatter funcion w/ logistic regression
# plt.scatter(x, y)
# plt.plot(X_test, loss, color="red", linewidth=3)
# plt.LogR()
# plt.xlabel("IHED")
# plt.ylabel("MOS expert")
# plt.show()
