import numpy as np
import pandas as pd
import lightgbm as lgbm
import random
import re

random.seed(5)
DATA_PATH = '/raid/aiqy/EGAM/output/Lakshimi_data/electronics/min_count5/'
EXP_FILE_ATT = DATA_PATH + 'drem-attention/explanation-features.csv'
#EXP_FILE_DREM = '/raid/aiqy/EGAM/output/Lakshimi_data/electronics/min_count5/drem-attention/explanation-features.csv'
EXP_FILE_DREM = DATA_PATH + '/drem/explanation-features.csv'
LABEL_FILE = DATA_PATH + 'all_samples_analysis.txt.updated.csv'

# ---------------------------------------
# Create Label Files

# process raw label files and create labels
heads = None
raw_label_data = {}
with open(LABEL_FILE) as fin:
    for line in fin:
        if heads == None:
            heads = line.strip().split('\t')
        else:
            arr = line.strip().split('\t')
            sample_id = arr[0]
            if sample_id not in raw_label_data:
                raw_label_data[sample_id] = []
            raw_label_data[sample_id].append([float(x) for x in arr[1:]])

label_map = {
    'usefulness' : {},
    'information' : {},
    'satisfaction' : {},
    'att_MRR' : {},
    'drem_MRR' : {}
}
def scores_to_label(Att, Drem): # x = group A - group B
    if Att == Drem:
        return 0
    elif Att > Drem: 
        return 1
    else: # B > A
        return -1

indexes = []
for sample_id in raw_label_data:
    indexes.append(sample_id)
    att_usefulness, att_information, att_satisfaction = 0,0,0
    drem_usefulness, drem_information, drem_satisfaction = 0,0,0
    for x in raw_label_data[sample_id]:
        att_usefulness += x[0]
        att_information += x[1]
        att_satisfaction += x[2]
        drem_usefulness += x[3]
        drem_information += x[4]
        drem_satisfaction += x[5]
    label_map['usefulness'][sample_id] = scores_to_label(att_usefulness, drem_usefulness)
    label_map['information'][sample_id] = scores_to_label(att_information, drem_information)
    label_map['satisfaction'][sample_id] = scores_to_label(att_satisfaction, drem_satisfaction)
    label_map['att_MRR'][sample_id] = raw_label_data[sample_id][0][6]
    label_map['drem_MRR'][sample_id] = raw_label_data[sample_id][0][7]

# ---------------------------------------
# Create Feature Files

ID_COLUMN_NAME = 'sample_id'
# features from EXP_FILES
#def rename_feature_columns(data, prefix):
#    data.columns = [x if x!= ID_COLUMN_NAME else prefix + x for x in data.columns]

att_features = pd.read_csv(EXP_FILE_ATT, sep='\t')
att_features = att_features.set_index(ID_COLUMN_NAME)
#rename_feature_columns(att_features, 'ATT_')
drem_features = pd.read_csv(EXP_FILE_DREM, sep='\t')
drem_features = drem_features.set_index(ID_COLUMN_NAME)

# performance features from LABEL_FILE
att_perf_features = {
    ID_COLUMN_NAME: indexes,
    'MRR': [label_map['att_MRR'][sample_id] for sample_id in indexes]
}
att_pf_features = pd.DataFrame(
    att_perf_features, 
    columns = [ID_COLUMN_NAME, 'MRR']
)
att_pf_features = att_pf_features.set_index(ID_COLUMN_NAME)
att_features = att_features.join(att_pf_features, how='inner')

drem_perf_features = {
    ID_COLUMN_NAME: indexes,
    'MRR': [label_map['drem_MRR'][sample_id] for sample_id in indexes]
}
drem_pf_features = pd.DataFrame(
    drem_perf_features, 
    columns = [ID_COLUMN_NAME, 'MRR']
)
drem_pf_features = drem_pf_features.set_index(ID_COLUMN_NAME)
drem_features = drem_features.join(drem_pf_features, how='inner')

#def create_feature_data(features_1, features_2):
#    features = features_1.join(features_2, how='inner', rsuffix='_other')
#    features = features.join(pf_features, how='inner')

# create train/test data
train_features_att = att_features.join(drem_features, how='inner', rsuffix='_other')
train_features_drem = drem_features.join(att_features, how='inner', rsuffix='_other') # switch feature position
feature_column_names = list(train_features_att.columns.values) 

# Append labels
att_label_map = {
    ID_COLUMN_NAME: indexes,
    'usefulness' : [1+label_map['usefulness'][sample_id] for sample_id in indexes],
    'information' : [1+label_map['information'][sample_id] for sample_id in indexes],
    'satisfaction' : [1+label_map['satisfaction'][sample_id] for sample_id in indexes]
}
att_labels = pd.DataFrame(
    att_label_map, 
    columns = [ID_COLUMN_NAME, 'usefulness', 'information', 'satisfaction']
)
att_labels = att_labels.set_index(ID_COLUMN_NAME)
train_features_att = train_features_att.join(att_labels, how='inner')

drem_label_map = {
    ID_COLUMN_NAME: indexes,
    'usefulness' : [1-label_map['usefulness'][sample_id] for sample_id in indexes],
    'information' : [1-label_map['information'][sample_id] for sample_id in indexes],
    'satisfaction' : [1-label_map['satisfaction'][sample_id] for sample_id in indexes]
}
drem_labels = pd.DataFrame(
    drem_label_map, 
    columns = [ID_COLUMN_NAME, 'usefulness', 'information', 'satisfaction']
)
drem_labels = drem_labels.set_index(ID_COLUMN_NAME)
train_features_drem = train_features_drem.join(drem_labels, how='inner')

# append training data together
final_train_data = train_features_att.append(train_features_drem)
#final_train_data = train_features_att
final_train_data.to_csv(DATA_PATH + 'all_explanation_feature_data.csv', sep='\t')  
# ---------------------------------------
# Cross Validation with lightgbm

label_names = ['usefulness', 'information', 'satisfaction']
N_FOLDS = 5
# build folds
data_folds = []
random.shuffle(indexes)
fold_size = int(len(indexes)/N_FOLDS)
for i in range(N_FOLDS):
    sub_indexes = indexes[i*fold_size:(i+1)*fold_size] if i<N_FOLDS-1 else indexes[i*fold_size:]
    data_folds.append(final_train_data.loc[sub_indexes])

# run cross validation
for cur_label_name in label_names:
    perf_map = {
        'Total_num' : [],
        'Correct' : [],
        'Type-1-error': [],
        'Type-2-error': [],
        'Type-n1-error': [],
        'Type-n2-error': [],
        'feature_importance' : []
    }

    for i in range(N_FOLDS):
        # split train/valid/test
        #cur_label_name = 'information'
        print('----------------------------\n Run Fold %d' % i)
        x_test = data_folds[i].drop(label_names, axis=1)
        x_test = x_test.values.astype(np.float32, copy=False)
        y_test = data_folds[i][cur_label_name].values
        #x_valid = data_folds[i-1].drop(label_names, axis=1)
        #x_valid = x_valid.values.astype(np.float32, copy=False)
        #y_valid = data_folds[i-1][cur_label_name].values
        d_valid = lgbm.Dataset(x_test, label=y_test)
        train_data = data_folds[i-1]
        for j in range(N_FOLDS-2):
            train_data = train_data.append(data_folds[i-j-2])
        x_train = train_data.drop(label_names, axis=1)
        x_train = x_train.values.astype(np.float32, copy=False)
        y_train = train_data[cur_label_name].values
        d_train = lgbm.Dataset(x_train, label=y_train)
        #print('Train %d, Valid %d, Test %d' % (len(y_train), len(y_valid), len(y_test)))
        print('Train %d, Test %d' % (len(y_train), len(y_test)))
        # set params for lightgbm
        params = {}
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'multiclassova'
        params['num_class'] = 3
        params['max_depth'] = 10 # 5-20
        params['num_leaves'] = 16 #10-30
        params['learning_rate'] = 0.3 # 0.1-0.5
        params['min_data_in_leaf'] = 45 # 10 - 50
        #params['tree_learner'] = 'voting_parallel'
        params['bagging_fraction'] = 1.0 # 0.1 -1.0
        params['num_iterations'] = 200
        params['feature_fraction'] = 0.5 # 0.1 - 1.0
        params['metric'] = 'multi_error'          # or 'mae'
        params['verbose'] = -2

        # train model
        clf = lgbm.train(params, d_train, valid_sets=d_valid, early_stopping_rounds=40) 

        # test model
        y_pred = clf.predict(x_test)
        y_pred_label = np.argmax(y_pred, axis=1)

        # evaluate test results
        perf_map['Total_num'].append(len(y_pred_label))
        correct, error_1, error_n1, error_2, error_n2 = 0, 0, 0, 0, 0
        for k in range(perf_map['Total_num'][-1]):
            error = y_pred_label[k] - y_test[k]
            if error == 0:
                correct += 1
            elif error == 1:
                error_1 += 1
            elif error == 2 :
                error_2 += 1
            elif error == -1 :
                error_n1 += 1
            elif error == -2 :
                error_n2 += 1
        perf_map['Correct'].append(correct)
        perf_map['Type-1-error'].append(error_1)
        perf_map['Type-2-error'].append(error_2)
        perf_map['Type-n1-error'].append(error_n1)
        perf_map['Type-n2-error'].append(error_n2)
        importance_scores = clf.feature_importance(importance_type='gain')
        importance_scores = importance_scores/np.sum(importance_scores) * 100
        perf_map['feature_importance'].append(importance_scores)
        #print(importance_scores)
        print('Total %d, Correct %d, Type-1-error %d, Type-2-error %d, Type-n1-error %d, Type-n2-error %d' % (len(y_pred_label), correct, error_1, error_2, error_n1, error_n2))

    # output results to csv
    #perf_map['Fold'] = [_ for _ in range(N_FOLDS)]
    pd.DataFrame(
        perf_map, 
        columns = ['Fold', 'Total_num','Correct','Type-1-error','Type-2-error','Type-n1-error','Type-n2-error']
    ).to_csv(DATA_PATH + '%d_cv_%s_results.csv' % (N_FOLDS, cur_label_name), sep='\t')  
    print(np.sum(perf_map['Correct']))
    #print(np.mean(np.array(perf_map['feature_importance']), axis=0))

    # ---------------------------------------
    # Feature Analysis
    feature_names = final_train_data.drop(label_names, axis=1).columns
    feat_impo_map = {}
    for im_scores in perf_map['feature_importance']:
        tmp_map = {}
        for importance, name in zip(im_scores, feature_names):
            name = re.sub('exp_[0-9]_', '', name)
            name = re.sub('_other', '', name)
            name = re.sub('_mean', '', name)
            name = re.sub('_min', '', name)
            name = re.sub('_max', '', name)
            if name not in tmp_map:
                tmp_map[name] = 0
            tmp_map[name] += importance
        for name in tmp_map:
            if name not in feat_impo_map:
                feat_impo_map[name] = []
            feat_impo_map[name].append(tmp_map[name])    
        #print('%s %.3f' % (name, importance) + '%')

    pd.DataFrame(
        feat_impo_map, 
        columns = list(feat_impo_map.keys()),
    ).to_csv(DATA_PATH + '%d_cv_%s_feature_importance.csv' % (N_FOLDS, cur_label_name), sep='\t')  
#print(feat_impo_map)

'''
x_train = final_train_data.drop(label_names, axis=1)
x_train = x_train.values.astype(np.float32, copy=False)

y_train = final_train_data['information'].values
d_train = lgbm.Dataset(x_train, label=y_train)
params = {}
params['boosting_type'] = 'gbdt'
params['objective'] = 'multiclass'
params['num_class'] = 3
params['metric'] = 'multi_error'          # or 'mae'
params['verbose'] = 0
cv_results = lgbm.cv(params, d_train, nfold=N_FOLDS, 
                    verbose_eval=50, early_stopping_rounds=40)



for cur_label_name in label_names:
    print('Classficiation for %s' % cur_label_name)
    
    # build dataset
    y_train = final_train_data[cur_label_name].values
    d_train = lgbm.Dataset(x_train, label=y_train)

    # set lightgbm params
    params = {}
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'multiclass'
    params['num_class'] = 3
    params['metric'] = 'multi_error'          # or 'mae'
    params['verbose'] = 0

    # run cross validation
    cv_results = lgbm.cv(params, d_train, nfold=N_FOLDS, 
                    verbose_eval=20, early_stopping_rounds=10, return_cvbooster=True)

    # output feature weights

'''












