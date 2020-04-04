# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
## Import Libraries

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import xgboost as xgb 
from xgboost import XGBClassifier
from sklearn import preprocessing
#sns.set(style = "whitegrid", color_codes = True)
os.chdir('D:/Study/Python Programs/DDT Interview Practice/')


# %%
## Data Preprocessing

# read data
data = pd.read_csv('speeddating.csv')

# rename colunns
data = data.rename(columns={'ambition':'ambitious', "ambitous_o":"ambitious_o", 'sinsere_o':'sincere_o', 'intellicence_important':'intelligence_important', 'ambtition_important':'ambitious_important', 'ambition_partner':'ambitious_partner','d_sinsere_o':'d_sincere_o', 'd_ambitous_o':'d_ambitious_o', 'd_intellicence_important':'d_intelligence_important', 'd_ambtition_important':'d_ambitious_important', 'd_ambition':'d_ambitious', 'd_ambition_partner':'d_ambitious_partner'})

data = data.replace('?', 0)


# %%
## Extract Interest Data
interests = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga']
interest_data = data[interests]
# data = data.drop(interests, axis=1)


# %%
def SplitAgeRange(age_range):
    age1, age2 = age_range[1:-1].split('-')
    return int(age1), int(age2)


# %%
data['samerace'] = data['samerace'] == 0


# %%
# Split Data by Rating Targets
features = ['attractive','sincere','intelligence','funny','ambitious', 'shared_interests']
own_feature_data = data[features[:-1]] # rate themselves
partner_rating_data = data[[f'{f}_o' for f in features]] # rated by partner
feature_importance_data = data[[f'{f}_important' for f in features]] # ideal partner
partner_feature_data = data[[f'{f}_partner' for f in features]] # rate partner


# %%
# Split Data by Feature Types
#attractive
attractive_data = data[['attractive', 'attractive_o', 'attractive_partner']].astype(float)
attractive_data = attractive_data[(attractive_data != 0).all(1)]
#sincere
sincere_data = data[['sincere', 'sincere_o', 'sincere_partner']].astype(float)
sincere_data = sincere_data[(sincere_data != 0).all(1)]
#intelligence
intelligence_data = data[['intelligence', 'intelligence_o', 'intelligence_partner']].astype(float)
intelligence_data = intelligence_data[(intelligence_data != 0).all(1)]

"""
attractive_data = data[['attractive', 'attractive_o', 'attractive_partner']].astype(float)
attractive_data = attractive_data[(attractive_data != 0).all(1)]

attractive_data = data[['attractive', 'attractive_o', 'attractive_partner']].astype(float)
attractive_data = attractive_data[(attractive_data != 0).all(1)]

attractive_data = data[['attractive', 'attractive_o', 'attractive_partner']].astype(float)
attractive_data = attractive_data[(attractive_data != 0).all(1)]
"""


# %%
#print(attractive_data['attractive'].astype(float).mean())
#print(attractive_data['attractive_o'].astype(float).mean())
#print(attractive_data['attractive_partner'].astype(float).mean())

#attractive_data.astype(float).boxplot()
fig, axs = plt.subplots(2,3)
#-attractive
axs[0,0].boxplot(attractive_data['attractive'])
axs[0,0].set_title('see themselves attractive')

axs[0,1].boxplot(attractive_data['attractive_o'])
axs[0,1].set_title('partner rate them attractive')

axs[0,2].boxplot(attractive_data['attractive_partner'])
axs[0,2].set_title('see partner attractive')
#-sincere
axs[1,0].boxplot(sincere_data['sincere'])
axs[1,0].set_title('see themselves sincere')

axs[1,1].boxplot(sincere_data['sincere_o'])
axs[1,1].set_title('partner rate them sincere')

axs[1,2].boxplot(sincere_data['sincere_partner'])
axs[1,2].set_title('see partner sincere')

fig.subplots_adjust(left = 0.02, right = 0.98, bottom = 0.1, top = 0.9, hspace = 0.6, wspace = 0.5)
"""
Min_Max_Scaler = preprocessing.MinMaxScaler( feature_range=(0,1) ) # 設定縮放的區間上下限
new_data = Min_Max_Scaler.fit_transform(attractive_data) # Data 為原始資料

new_data = pd.DataFrame(data = new_data, columns = ['attractive', 'attractive_o', 'attractive_important', 'attractive_partner'])
new_data.boxplot()
"""


# %%
# Pearson Correlation Matrix (Ideal Partner)
corr_columns = ['importance_same_race', 'importance_same_religion', 'attractive_important', 'sincere_important', 'intelligence_important','funny_important', 'ambitious_important', 'shared_interests_important', 'interests_correlate', 'match']
corr_table = data[corr_columns[:-1]]
corr_table = corr_table.rename(columns={'importance_same_race':'same_race','importance_same_religion':'same_religion','attractive_important':'attractive', "sincere_important":"sincere", 'intelligence_important':'intelligence', 'funny_important':'funny', 'ambitious_important':'ambitious', 'shared_interests_important':'shared_interests'})
corr_table = corr_table[(corr_table != 0).all(1)] #delete rows with all rating zero
corr_table = corr_table.astype(float)
pearsoncorr = corr_table.corr(method = 'pearson')
pearsoncorr


# %%
# Plot HeatMap of Correlation Matrix
plt.figure(figsize = (10,6))
plt.title('Correlation Matrix - Ideal Partner Features')
sns.heatmap(data = pearsoncorr, annot = True, fmt = '.0%', cmap = 'YlGnBu')
plt.show()


# %%
# create clean_data table for xgboost model
clean_data = data.drop(['d_expected_num_interested_in_me', 'd_expected_num_matches', 'd_like', 'd_guess_prob_liked', 'd_expected_happy_with_sd_people', 'd_interests_correlate', 'd_d_age', 'd_importance_same_religion', 'd_importance_same_race'], axis=1)
clean_data = clean_data.drop([f'd_pref_o_{f}' for f in features], axis=1)
clean_data = clean_data.drop([f'd_{f}_o' for f in features], axis=1)
clean_data = clean_data.drop([f'd_{f}_important' for f in features], axis=1)
clean_data = clean_data.drop([f'd_{f}' for f in features[:-1]], axis=1)
clean_data = clean_data.drop([f'd_{f}_partner' for f in features], axis = 1)
clean_data = clean_data.drop([f'd_{i}' for i in interests], axis = 1)
clean_data = clean_data.replace('?', 0)
clean_data['user'] = clean_data.groupby(features[:-1]).grouper.group_info[0] + 1


# %%
matched_cor = clean_data[['user', 'interests_correlate']].astype(float)[clean_data['match'] == 0].groupby('user').mean()


# %%
total_cor = clean_data[['user', 'interests_correlate']].astype(float).groupby('user').mean()


# %%
(total_cor.mean() - matched_cor.mean()) / np.std(clean_data['interests_correlate'].astype(float))


# %%
np.std(clean_data['interests_correlate'].astype(float))


# %%
# selected parameters, drop parameters (including drop_fileds and not_suitables)
param = {'booster':'gblinear', 'lambda':1, 'alpha':0, 'subsample':1, 'predictor':'cpu_predictor', 'max_depth':20}
drop_fields = ['match', 'race', 'race_o', 'gender', 'field', 'decision', 'decision_o', 'like', 'guess_prob_liked', 'has_null', 'wave', 'user']
not_suitables = ['shopping', 'tv', 'shared_interests_o', 'attractive_o']

drop_fields.extend(not_suitables)
#drop_fields.extend(clean_data.columns[5:])


accuracy_1 = 0
accuracy_0 = 0
model_features = []

ac0 = 0
for i in range(10):
    testing = clean_data.sample(frac=1/5)
    training = clean_data[(~clean_data.isin(testing))].dropna() 
    
    for _ in range(5):
        #testing = testing.append(testing[testing['match'] == 1])
        training = training.append(training[training['match'] == 1])

    model = XGBClassifier()
    model.fit(training.drop(drop_fields, axis=1).astype(float), training['match'].astype(float))

    ypred = model.predict(testing.drop(drop_fields, axis=1).astype(float))

    #ac1 = sum(ypred == testing['match']) / testing.shape[0]
    ac1 = sum((ypred == testing['match'])[testing['match'] == 1]) / testing[testing['match'] == 1].shape[0]
    ac0 = sum((ypred == testing['match'])[testing['match'] == 0]) / testing[testing['match'] == 0].shape[0]
    print(ac1, ac0)

    model_features.append(model.feature_importances_)
    accuracy_1 += ac1
    accuracy_0 += ac0

print(f'Average 1: {accuracy_1 / 10}')
print(f'Average 0: {accuracy_0 / 10}')
print((accuracy_0 + accuracy_1)/ 20)
# dTrain = xgb.DMatrix(own_feature_data.astype(float), label=clean_data['match'].astype(float))
# bst = xgb.train(param, dTrain) #, xgb_model=bst)  # for retraining
# bst.save_model('./xgb_model.model')


# %%
print([np.mean(n) for n in np.array(model_features).swapaxes(0,1)] * np.array(1))
print([np.std(n) / np.mean(n)for n in np.array(model_features).swapaxes(0,1)] * np.array(len(model.feature_importances_)))


# %%
clean_data.drop(drop_fields, axis=1).columns[12]


# %%
sum(clean_data['match'] == 0)


# %%
testing.shape


# %%


