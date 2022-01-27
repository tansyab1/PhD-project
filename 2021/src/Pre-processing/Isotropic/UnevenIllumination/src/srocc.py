import pandas as pd
import numpy as np
from scipy import stats

df_sdv = pd.read_csv(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/sdv2.csv', delimiter=',', header=None)

df_agic = pd.read_csv(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/agic.csv', delimiter='\t', header=None)

df_ihed = pd.read_csv(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/ihed2.csv', delimiter=',', header=None)

df_mosEx = pd.read_csv(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/mosEx.csv', delimiter=' ', header=None)

df_mosNonEx = pd.read_csv(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/mosNonEx.csv', delimiter=' ', header=None)

# estimated coefficients of different models
df_ihed.columns = ['file', 'value']
df_agic.columns = ['file', 'value']
df_sdv.columns = ['file', 'value']

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

# Calcalate the Spearman correlation coefficient
print("SROCC between each measure and MOS expert:")

print("Spearman correlation coefficient between SDV and MOS expert==>",
      stats.spearmanr(sorted_sdv, sorted_mosEx))
print("Spearman correlation coefficient between AGIC and MOS expert==>",
      stats.spearmanr(sorted_agic, sorted_mosEx))
print("Spearman correlation coefficient between IHED and MOS expert==>",
      stats.spearmanr(sorted_ihed, sorted_mosEx))

print("SROCC between each measure and MOS non-expert:")
print("Spearman correlation coefficient between SDV and MOS non-expert==>",
      stats.spearmanr(sorted_sdv, sorted_mosNonEx))
print("Spearman correlation coefficient between AGIC and MOS non-expert==>",
      stats.spearmanr(sorted_agic, sorted_mosNonEx))
print("Spearman correlation coefficient between IHED and MOS non-expert==>",
      stats.spearmanr(sorted_ihed, sorted_mosNonEx))


# Calcalate the Pearson correlation coefficient
print("LCC between each measure and MOS expert:")
print("Pearson correlation coefficient between SDV and MOS expert==>",
      stats.pearsonr(sorted_sdv, sorted_mosEx))
print("Pearson correlation coefficient between AGIC and MOS expert==>",
      stats.pearsonr(sorted_agic, sorted_mosEx))
print("Pearson correlation coefficient between IHED and MOS expert==>",
      stats.pearsonr(sorted_ihed, sorted_mosEx))

print("LCC between each measure and MOS non-expert:")
print("Pearson correlation coefficient between SDV and MOS non-expert==>",
      stats.pearsonr(sorted_sdv, sorted_mosNonEx))
print("Pearson correlation coefficient between AGIC and MOS non-expert==>",
      stats.pearsonr(sorted_agic, sorted_mosNonEx))
print("Pearson correlation coefficient between IHED and MOS non-expert==>",
      stats.pearsonr(sorted_ihed, sorted_mosNonEx))
