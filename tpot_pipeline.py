import pandas as pd
import numpy as np
import gc
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns

from category_encoders import WOEEncoder
from tpot import TPOTClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import LinearRegression
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RepeatedKFold

""" Feature engineering """

at_df = pd.read_csv('application_train.csv')
at_df.drop(at_df.columns[at_df.isnull().sum() > 173378], axis=1, inplace=True)

# fix day counters which are negative
at_df['DAYS_BIRTH'] = at_df['DAYS_BIRTH'] / -356
at_df['DAYS_EMPLOYED'] = at_df['DAYS_EMPLOYED'] / -356
at_df['DAYS_REGISTRATION'] = at_df['DAYS_REGISTRATION'] / -356
at_df['DAYS_ID_PUBLISH'] = at_df['DAYS_ID_PUBLISH'] / -356
at_df.rename(columns={'DAYS_BIRTH': 'YEARS_BIRTH',  'DAYS_EMPLOYED': 'YEARS_EMPLOYED', 'DAYS_REGISTRATION': 'YEARS_REGISTRAION', 'DAYS_ID_PUBLISH': 'YEARS_ID_PUBLISH'}, inplace=True)


b_df = pd.read_csv('bureau.csv')
bb_df = pd.read_csv('bureau_balance.csv')

# extract more value from the status column
bb_df['STATUS'].replace({'X': -2, 'C':-1}, inplace=True)
bb_df['STATUS'] = pd.to_numeric(bb_df['STATUS'])

# read in more data
pa_df = pd.read_csv('Previous_Application.csv')
ccb_df = pd.read_csv('credit_card_balance.csv')
pcb_df = pd.read_csv('POS_CASH_balance.csv')
ip_df = pd.read_csv('installments_payments.csv')

# engineer aggregate statistics and merge
bb_agg = bb_df.groupby('SK_ID_BUREAU')['STATUS'].agg(['mean', 'median', 'max', 'min', 'count'])
b_df = b_df.merge(bb_agg, on='SK_ID_BUREAU', how='left')

# separate numeric and categorical data and again computer aggregates
b_c = b_df.select_dtypes(exclude=[np.number])
b_c = pd.get_dummies(b_c)
b_c['SK_ID_CURR'] = b_df['SK_ID_CURR']
b_c = b_c.groupby('SK_ID_CURR').mean()
b_c = b_c[b_c.columns[(b_c.mean() > 0.001)]]
b_df = b_df.select_dtypes(include=[np.number])
b_agg = b_df.drop(columns=['SK_ID_BUREAU']).groupby('SK_ID_CURR').agg(['mean', 'median', 'min', 'max', 'sum', 'count'])

# combine aggregates
b_agg.columns = b_agg.columns.to_flat_index()
b_agg = b_agg.merge(b_c, on='SK_ID_CURR', how='left')
b_agg.reset_index(inplace=True)
b_agg.columns = b_agg.columns.map(str)

# drop columns with too many NA's and then drop mostly null rows
pa_df.drop(columns=pa_df.columns[pa_df.isnull().sum() > 600000], axis=1, inplace=True)
pa_df.drop(columns=['NFLAG_LAST_APPL_IN_DAY'], inplace=True)

# separate categorical and numeric data and compute aggregates
pa_c = pa_df.select_dtypes(exclude=[np.number])
pa_df = pa_df.select_dtypes(include=[np.number])

pa_agg = pa_df.drop(columns=['SK_ID_PREV']).groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'median', 'sum', 'count'])
pa_agg.columns = pa_agg.columns.to_flat_index()
pa_agg.rename(columns={'(SK_ID_CURR, )': 'SK_ID_CURR'}, inplace=True) 
pa_c = pd.get_dummies(pa_c)
pa_c['SK_ID_CURR'] = pa_df['SK_ID_CURR']
pa_c = pa_c.groupby('SK_ID_CURR').mean()
pa_c = pa_c[pa_c.columns[(pa_c.mean() > 0.1)]]

# combine aggregates
pa_agg = pa_agg.merge(pa_c, on='SK_ID_CURR', how='left')
pa_agg.columns = pa_agg.columns.map(str)

# credit card balance
ccb_df.drop(columns=['NAME_CONTRACT_STATUS'], inplace=True)
ccb_agg = ccb_df.drop(columns=['SK_ID_PREV', 'MONTHS_BALANCE']).groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'median', 'sum', 'count'])
ccb_agg.columns = ccb_agg.columns.to_flat_index()

# POS CASH
pcb_agg = pcb_df.drop(columns=['MONTHS_BALANCE', 'NAME_CONTRACT_STATUS', 'SK_ID_PREV']).groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'median', 'sum', 'count'])
pcb_agg.columns = pcb_agg.columns.to_flat_index()

# Installments Payments
ip_agg = ip_df.drop(columns=['SK_ID_PREV']).groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'median', 'sum', 'count'])
ip_agg.columns = ip_agg.columns.to_flat_index()

# combine aggregates from previous applications
pa_agg = pa_agg.merge(ccb_agg, on='SK_ID_CURR', how='left')
pa_agg = pa_agg.merge(pcb_agg, on='SK_ID_CURR', how='left')
pa_agg = pa_agg.merge(ip_agg, on='SK_ID_CURR', how='left')

# now merge all selected features into a single dataframe
selected = at_df.merge(pa_agg, on='SK_ID_CURR', how='left')
selected = selected.merge(b_agg, on='SK_ID_CURR', how='left')
selected.columns = selected.columns.map(str)

# engineer some additional features that are likely to improve the model
selected['EXT_SOURCE_COMB1'] = selected['EXT_SOURCE_1'] * selected['EXT_SOURCE_2']
selected['EXT_SOURCE_COMB2'] = selected['EXT_SOURCE_1'] * selected['EXT_SOURCE_3']
selected['EXT_SOURCE_COMB3'] = selected['EXT_SOURCE_2'] * selected['EXT_SOURCE_3']
selected['EXT_SOURCE_1_SQR'] = selected['EXT_SOURCE_1'] * selected['EXT_SOURCE_1']
selected['EXT_SOURCE_2_SQR'] = selected['EXT_SOURCE_2'] * selected['EXT_SOURCE_2']
selected['EXT_SOURCE_3_SQR'] = selected['EXT_SOURCE_3'] * selected['EXT_SOURCE_3']

selected['CREDIT_OVER_INCOME'] = selected['AMT_CREDIT'] / selected['AMT_INCOME_TOTAL']
selected['AGE_OVER_CHILDREN'] = selected['YEARS_BIRTH'] / (1 + selected['CNT_CHILDREN'])
selected['CREDIT_LENGTH'] = selected['AMT_ANNUITY'] * selected['AMT_CREDIT']
selected['STABILITY'] = selected['YEARS_EMPLOYED'] * selected['REGION_RATING_CLIENT']

# drop rows with excessive NA's
selected = selected[selected.isnull().sum(axis=1) < selected.isnull().sum(axis=1).mean() * 0.75]

# check feature correlations with target
cors = selected.corr()
tcors = cors['TARGET']
tcors.fillna(0, inplace=True)


# train test split
labels = selected['TARGET']
ids = selected['SK_ID_CURR']
selected.drop(columns=['TARGET', 'SK_ID_CURR'], inplace=True)
trainX, testX, trainY, testY = train_test_split(selected, labels, test_size=0.25, shuffle=True)

# seperate numeric and categoric data
trainXn = trainX.select_dtypes(include=[np.number])
trainXc = trainX.select_dtypes(exclude=[np.number])
testXn = testX.select_dtypes(include=[np.number])
testXc = testX.select_dtypes(exclude=[np.number])

# collect garbage
mem_check = False
if mem_check:
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']
    print(sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True))

d = True
if d:
    del [[at_df, b_df, bb_df, pa_df, ccb_df, pcb_df, ip_df, bb_agg, b_agg, pa_agg, ccb_agg, pcb_agg, ip_agg, b_c, pa_c, selected, labels]]
    gc.collect()
    at_df = pd.DataFrame()
    b_df = pd.DataFrame()
    ccb_df = pd.DataFrame()
    b_c = pd.DataFrame()
    pa_c = pd.DataFrame()
    pcb_df = pd.DataFrame()
    ip_df = pd.DataFrame()
    b_agg = pd.DataFrame()
    bb_df = pd.DataFrame()
    bb_agg = pd.DataFrame()
    pa_df = pd.DataFrame()
    pa_agg = pd.DataFrame()
    ccb_agg = pd.DataFrame()
    pcb_agg = pd.DataFrame()
    ip_agg = pd.DataFrame()
    selected = pd.DataFrame()

# impute missing/null values
l = LinearRegression()
i_imputer = IterativeImputer(estimator=l, add_indicator=True, max_iter=5)
trainXn = pd.DataFrame(i_imputer.fit_transform(trainXn))
testXn = pd.DataFrame(i_imputer.transform(testXn))

# standardize data
r_scaler = RobustScaler()
trainXn = pd.DataFrame(r_scaler.fit_transform(trainXn))
testXn = pd.DataFrame(r_scaler.transform(testXn))

# deploy weight of evidence encoding to assign numeric values to categorical data
woe = WOEEncoder()
trainXc = woe.fit_transform(trainXc, trainY)
testXc = woe.transform(testXc)

s_imputer = SimpleImputer(strategy='median', add_indicator=True)
trainXc = pd.DataFrame(s_imputer.fit_transform(trainXc))
testXc = pd.DataFrame(s_imputer.transform(testXc))

new_cols = []
for i in range(len(testXc.columns)):
    new_cols.append(str(trainXc.columns[i] + trainXn.shape[1]))
trainXc.columns = new_cols
testXc.columns = new_cols

# combine processed to to obtain final features
frames = [trainXn.reset_index(drop=True), trainXc.reset_index(drop=True)]
trainX = pd.concat(frames, axis=1)
frames = [testXn.reset_index(drop=True), testXc.reset_index(drop=True)]
testX = pd.concat(frames, axis=1)

del [[trainXn, trainXc, testXn, testXc]]
gc.collect()
trainXn = pd.DataFrame()
trainXc = pd.DataFrame()
testXn = pd.DataFrame()
testXc = pd.DataFrame()



""" Model Selection & Hyperparameter Optimization """

cv = RepeatedKFold(n_splits=10, n_repeats=3)
tpot_clf = TPOTClassifier(generations=250, population_size=12, verbosity=2, scoring='f1', warm_start=True, n_jobs=-1)

tpot_clf.fit(trainX, trainY)
tpot_score = tpot_clf.score(testX, testY)

try:
    probs = tpot_clf.predict_proba(testX)
    preds = probs.argmax(axis=1)
except:
    preds = tpot_clf.predict(testX)

tpot_clf.export('tpot')

tn, fp, fn, tp = confusion_matrix(testY, preds).ravel()

B = 1 - testY.sum()/testY.count()
accuracy = (tn + tp) / (tn + fp + tp + fn + 0.0001)
precision = tp / (tp + fp + 0.0001)
recall = tp / (tp + fn + 0.0001)
f1 = (1 + B ** 2) * (precision * recall) / (B ** 2 * precision + recall + 0.0001)
print(accuracy, precision, recall, f1)

cm = confusion_matrix(testY, preds)
ax = sns.heatmap(cm, annot=True, cmap="PuBuGn", fmt='g')
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

fpr, tpr, thresholds = roc_curve(testY, probs[:, 1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

print("AUC: ", roc_auc_score(testY, probs[:, 1]))

