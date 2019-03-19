# import basic packages
import numpy as np
# import self-defined modules
from implementations import *
from tools import *
from helpers import *
# just to ingore warning
import warnings
warnings.filterwarnings('ignore')

# -------------------- load train and test data -------------------- #
print('Start loading data...',end='')
data_path_tr = 'data/train.csv'
data_path_te = 'data/test.csv'
yb_tr, data_tr, idx_tr, labels = load_csv_data(data_path_tr, sub_sample=False)
yb_te, data_te, idx_te, _      = load_csv_data(data_path_te, sub_sample=False)

# prepare the data using self-defined DataFrame class
labels_dataframe = ['Prediction'] + labels
data_tr_dataframe = np.concatenate((yb_tr.reshape([-1, 1]), data_tr), axis=1)
data_te_dataframe = np.concatenate((yb_te.reshape([-1, 1]), data_te), axis=1)
dataframe_tr = DataFrame(data_tr_dataframe, idx_tr.tolist(), labels_dataframe)
dataframe_te = DataFrame(data_te_dataframe, idx_te.tolist(), labels_dataframe)
print('Completed')
# -------------------- data processing -------------------- #
# replace missing values with mode
print('Start processing data...',end='')
DER_mode_s_tr = 119.89
DER_mode_b_tr = 96.819
temp = dataframe_tr.loc(dataframe_tr['DER_mass_MMC']==-999)
temp = temp.loc(temp['Prediction']==1)
dataframe_tr.loc(temp.index)['DER_mass_MMC'] = DER_mode_s_tr
temp = dataframe_tr.loc(dataframe_tr['DER_mass_MMC']==-999)
temp = temp.loc(temp['Prediction']==-1)
dataframe_tr.loc(temp.index)['DER_mass_MMC'] = DER_mode_b_tr

DER_mode_te = 96.728
temp = dataframe_te.loc(dataframe_te['DER_mass_MMC']==-999)
dataframe_te.loc(temp.index)['DER_mass_MMC'] = DER_mode_te

# drop features
feature_dorp_phi = ['PRI_jet_leading_phi',
                    'PRI_jet_subleading_phi',
                    'PRI_lep_phi',
                    'PRI_met_phi',
                    'PRI_tau_phi']
dataframe_tr = dataframe_tr.drop(feature_dorp_phi)
dataframe_te = dataframe_te.drop(feature_dorp_phi)
# divide training data to 3 groups according to feature 'PRI_jet_num'
def group_features_by_jet(dataframe):
    return {
        0: dataframe.loc( dataframe['PRI_jet_num'] == 0).copy(),
        1: dataframe.loc( dataframe['PRI_jet_num'] == 1).copy(),
        2: dataframe.loc((dataframe['PRI_jet_num'] == 2) | (dataframe['PRI_jet_num'] == 3)).copy()}
dataframe_tr_grp = group_features_by_jet(dataframe_tr)
dataframe_te_grp = group_features_by_jet(dataframe_te)
# drop features with undefined values (features whose missing rate of -999 are 100%)
feature_undefined_gp0 = ['DER_deltaeta_jet_jet',
                         'DER_mass_jet_jet',
                         'DER_prodeta_jet_jet',
                         'DER_lep_eta_centrality',
                         'PRI_jet_leading_pt',
                         'PRI_jet_leading_eta',
                         'PRI_jet_subleading_pt',
                         'PRI_jet_subleading_eta']
feature_undefined_gp1 = ['DER_deltaeta_jet_jet',
                         'DER_mass_jet_jet',
                         'DER_prodeta_jet_jet',
                         'DER_lep_eta_centrality',
                         'PRI_jet_subleading_pt',
                         'PRI_jet_subleading_eta']
dataframe_tr_grp[0] = dataframe_tr_grp[0].drop(feature_undefined_gp0)
dataframe_tr_grp[1] = dataframe_tr_grp[1].drop(feature_undefined_gp1)
dataframe_te_grp[0] = dataframe_te_grp[0].drop(feature_undefined_gp0)
dataframe_te_grp[1] = dataframe_te_grp[1].drop(feature_undefined_gp1)
# drop feature 'PRI_jet_num' which is already used for grouping
dataframe_tr_grp[0] = dataframe_tr_grp[0].drop('PRI_jet_num')
dataframe_tr_grp[1] = dataframe_tr_grp[1].drop('PRI_jet_num')
dataframe_tr_grp[2] = dataframe_tr_grp[2].drop('PRI_jet_num')
dataframe_te_grp[0] = dataframe_te_grp[0].drop('PRI_jet_num')
dataframe_te_grp[1] = dataframe_te_grp[1].drop('PRI_jet_num')
dataframe_te_grp[2] = dataframe_te_grp[2].drop('PRI_jet_num')
# drop features with 0 values
dataframe_tr_grp[0] = dataframe_tr_grp[0].drop('PRI_jet_all_pt')
dataframe_te_grp[0] = dataframe_te_grp[0].drop('PRI_jet_all_pt')
# get the finally data
data_tr_grp = []
pred_tr_grp = []
for index in range(len(dataframe_tr_grp)):
    data_tr_grp.append(dataframe_tr_grp[index].drop('Prediction').values)
    pred_tr_grp.append((dataframe_tr_grp[index])['Prediction'])
data_te_grp = []
pred_te_grp = []
for index in range(len(dataframe_te_grp)):
    data_te_grp.append(dataframe_te_grp[index].drop('Prediction').values)
    pred_te_grp.append((dataframe_te_grp[index])['Prediction'])
print('Completed')
# -------------------- model training -------------------- #
# use ridge regression model
print('Start training...', end='')
lambda_ = 0.0005
degrees = [9, 9, 12]
k_fold = 10
seed = 10
y_te = []
w = []
for i in range(len(data_tr_grp)):
    # model training
    x = data_tr_grp[i]
    x = log_process(x)
    x, _, _ = standardize(x)
    x_tr = build_poly(x, degrees[i])
    y_tr = pred_tr_grp[i]
    w_tmp, _, _ = cv_loop(y_tr, x_tr, k_fold, seed, ridge_regression, lambda_=lambda_)
    w.append(w_tmp)
    # test model
    x = data_te_grp[i]
    x = log_process(x)
    x, _, _ = standardize(x)
    x_te = build_poly(x, degrees[i])
    y_te.append(predict_labels(w[i], x_te))

y_pred_te = np.concatenate((y_te[0], y_te[1], y_te[2]))
idx_te = np.concatenate((np.array(dataframe_te_grp[0].index),
                         np.array(dataframe_te_grp[1].index),
                         np.array(dataframe_te_grp[2].index)))
print('Completed')
# -------------------- generate prediction files -------------------- #
print('Start generating prediction files...', end='')
output_path = 'data/Predicion.csv'
create_csv_submission(idx_te, y_pred_te, output_path)
print('Completed')
