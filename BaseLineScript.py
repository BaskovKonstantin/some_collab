import sys
import os
import warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')
###################################################################################################################################################
import pandas as pd
import numpy as np
import time
import pyarrow.parquet as pq
import scipy
import implicit
import bisect
import sklearn.metrics as m
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
###################################################################################################################################################
LOCAL_DATA_PATH = ''
SPLIT_SEED = 42
DATA_FILE = 'competition_data_final_pqt'
TARGET_FILE = 'public_train.pqt'
SUBMISSION_FILE = 'submit.pqt'
###################################################################################################################################################
id_to_submit = pq.read_table(f'{SUBMISSION_FILE}').to_pandas()

data = pq.read_table(f'{DATA_FILE}')
pd.DataFrame([(z.name, z.type) for z in data.schema], columns = [['field', 'type']])
data.select(['cpe_type_cd']).to_pandas()['cpe_type_cd'].value_counts()
targets = pq.read_table(f'{TARGET_FILE}')
pd.DataFrame([(z.name, z.type) for z in targets.schema], columns = [['field', 'type']])

data_agg = data.select(['user_id', 'url_host', 'request_cnt']).\
    group_by(['user_id', 'url_host']).aggregate([('request_cnt', "sum")])

url_set = set(data_agg.select(['url_host']).to_pandas()['url_host'])
print(f'{len(url_set)} urls')
url_dict = {url: idurl for url, idurl in zip(url_set, range(len(url_set)))}
usr_set = set(data_agg.select(['user_id']).to_pandas()['user_id'])
print(f'{len(usr_set)} users')
usr_dict = {usr: user_id for usr, user_id in zip(usr_set, range(len(usr_set)))}
###################################################################################################################################################
values = np.array(data_agg.select(['request_cnt_sum']).to_pandas()['request_cnt_sum'])
rows = np.array(data_agg.select(['user_id']).to_pandas()['user_id'].map(usr_dict))
cols = np.array(data_agg.select(['url_host']).to_pandas()['url_host'].map(url_dict))
mat = scipy.sparse.coo_matrix((values, (rows, cols)), shape=(rows.max() + 1, cols.max() + 1))
als = implicit.approximate_als.FaissAlternatingLeastSquares(factors = 50, iterations = 30, use_gpu = False, \
       calculate_training_loss = False, regularization = 0.1)


als.fit(mat)
u_factors = als.model.user_factors
d_factors = als.model.item_factors

inv_usr_map = {v: k for k, v in usr_dict.items()}
usr_emb = pd.DataFrame(u_factors)
usr_emb['user_id'] = usr_emb.index.map(inv_usr_map)
usr_targets = targets.to_pandas()
df = usr_targets.merge(usr_emb, how = 'inner', on = ['user_id'])
df = df[df['is_male'] != 'NA']
df = df.dropna()
df['is_male'] = df['is_male'].map(int)
df['is_male'].value_counts()
###################################################################################################################################################
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

x_train, x_test, y_train, y_test = train_test_split(df.drop(['user_id', 'age', 'is_male'], axis=1), df['is_male'], test_size=0.33, random_state=SPLIT_SEED)

catboost_model = CatBoostClassifier()
naive_bayes_model = GaussianNB()
knn_model = KNeighborsClassifier()

estimators = [('catboost', catboost_model), ('naive_bayes', naive_bayes_model), ('knn', knn_model)]

voting_model = VotingClassifier(estimators=estimators, voting='soft')

param_grid = {
    'catboost__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'catboost__depth': [3, 5, 7, 10, 14],
    'naive_bayes__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
    'knn__n_neighbors': [3, 5, 7, 10, 14],
}

grid_search = GridSearchCV(voting_model, param_grid, cv=5)
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_

print(f'GINI по полу {2 * m.roc_auc_score(y_test, best_model.predict_proba(x_test)[:, 1]) - 1:2.3f}')


best_model.fit(df.drop(['user_id', 'age', 'is_male'], axis = 1), df['is_male'], verbose = False)
id_to_submit['is_male'] = best_model.predict_proba(id_to_submit.merge(usr_emb, how = 'inner', on = ['user_id']))[:,1]
###################################################################################################################################################
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

sns.set_style('darkgrid')

def age_bucket(x):
    return bisect.bisect_left([18,25,35,45,55,65], x)

df = usr_targets.merge(usr_emb, how = 'inner', on = ['user_id'])
df = df[df['age'] != 'NA']
df = df.dropna()
df['age'] = df['age'].map(age_bucket)
sns.histplot(df['age'], bins = 7)
###################################################################################################################################################
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from catboost import CatBoostClassifier

x_train, x_test, y_train, y_test = train_test_split(df.drop(['user_id', 'age', 'is_male'], axis=1), df['age'], test_size=0.33, random_state=SPLIT_SEED)

svm_model = SVC()
catboost_model = CatBoostClassifier()

estimators = [('svm', svm_model), ('catboost', catboost_model)]

stacking_model = StackingClassifier(estimators=estimators, final_estimator=SVC())

param_grid = {
    'svm__C': [0.1, 1, 10, 50, 100],
    'svm__gamma': [0.1, 1, 10, 50, 100],
    'final_estimator__C': [0.1, 1, 10, 50, 100],
    'final_estimator__gamma': [0.1, 1, 10, 50, 100]
}

grid_search = GridSearchCV(stacking_model, param_grid, cv=5)
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_

print(m.classification_report('SVC', y_test, best_model.predict(x_test), target_names=['<18', '18-25','25-34', '35-44', '45-54', '55-65', '65+']))
###################################################################################################################################################
best_model.fit(df.drop(['user_id', 'age', 'is_male'], axis = 1), df['age'], verbose = False)
id_to_submit['age'] = best_model.predict(id_to_submit[['user_id']].merge(usr_emb, how = 'inner', on = ['user_id']))
###################################################################################################################################################
id_to_submit.head()
id_to_submit.to_csv(f'{LOCAL_DATA_PATH}/submission.csv', index = False)
