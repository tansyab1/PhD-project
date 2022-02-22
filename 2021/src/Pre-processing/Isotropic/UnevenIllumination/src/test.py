
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

df_sdv = pd.read_csv(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/sdv2.csv', delimiter=',', header=None)

df_agic = pd.read_csv(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/agic.csv', delimiter='\t', header=None)

df_ihed = pd.read_csv(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/ihed_corected_done.csv', delimiter=',', header=None)

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


x = np.array(sorted_ihed, dtype=float)
y = np.array(sorted_mosEx, dtype=float)

X = x[:, np.newaxis]
print(y.shape)
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y)
# Fit the classifier
clf = LogisticRegression(C=10, solver='lbfgs')
print(training_scores_encoded)
clf.fit(X, training_scores_encoded)

# and plot the result
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.scatter(X.ravel(), training_scores_encoded, color="black", zorder=20)

x_range = 15000
Xs = [i*0.0001 for i in range(x_range)]
Ys = [clf.predict([[value*0.0001]]) for value in range(x_range)]
plt.plot(Xs, Ys, color='red')
# ols = LinearRegression()
# ols.fit(X, y)
# plt.plot(X_test, ols.coef_ * X_test + ols.intercept_, linewidth=1)
# plt.axhline(0.5, color=".5")

# plt.ylabel("y")
# plt.xlabel("X")
# plt.xticks(range(-5, 10))
# plt.yticks([0, 0.5, 1])
# plt.ylim(-0.25, 1.25)
# plt.xlim(-4, 10)
# plt.legend(
#     ("Logistic Regression Model", "Linear Regression Model"),
#     loc="lower right",
#     fontsize="small",
# )
# plt.tight_layout()
plt.show()