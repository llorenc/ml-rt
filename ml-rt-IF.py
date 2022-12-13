#!/usr/bin/python3 -d
# -*- coding: utf-8 -*-
# process ml-rt-training_dataset.pkl and ml-rt-testing_dataset.pkl with RT datasets
# (c) Llorenç Cerdà-Alabern, September 2022.
# debug: import pdb; pdb.set_trace()
# https://docs.h5py.org/en/stable/quick.html

# imports
import IPython
import sklearn as skl
import importlib
import os
import re
import numpy as np
import pandas as pd
import datetime as dt
from itertools import chain
import json
import click   ## https://click.palletsprojects.com/en/7.x
import fnmatch
from operator import itemgetter
import bz2
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as font_manager
plt.ion()  # interactive non-blocking mode
import time
import shap

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
# from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.lscp import LSCP

# pyod parameters
contamination_est = 0.005
# random_state = np.random.RandomState(42)
random_state  = 42
classifiers = {}
try:
    print("importing VAE")
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.models.vae import VAE
    classifiers.update({
        'Variational auto encoder (VAE)':
        VAE(encoder_neurons=[128, 64, 32], decoder_neurons=[32, 64, 128], 
            epochs=30, batch_size=32,
            dropout_rate=0.2,contamination=contamination_est, verbose=1)})
except:
    print("skipping VAE, tensorflow not installed")

# wd
pwd = os.getcwd()
print('pwd: ' + pwd)
if not os.path.exists("ml-rt.py"):
    wd = os.environ['HOME'] + '/recerca/connexio-guifinet/meshmon/py-nade/ml-rt'
    if os.path.exists(wd):
        print('wd: ' + wd)
        os.chdir(wd)

# local modules
sys.path.insert(1, '../ml')
imported = {}
def force_import(name):
    global imported
    if name not in imported:
        imported[name] = importlib.import_module(name)
    else:
        importlib.reload(imported[name])
    return imported[name]
uid = force_import("uid")

def say(msg):
    if not type(msg) is str:
        msg = str(msg)
    click.secho(msg, fg="green")

# Pickle a file and then compress it into a file with extension 
def compress_pickle(data, fname):
    if not re.search(r'bz2$', fname):
        fname = "{}.bz2".format(fname)
    with bz2.BZ2File(fname, 'w') as fh:
        pickle.dump(data, fh)
# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    return pickle.load(data)
# 
def load_and_compress_pickle(fn):
    zf = "{}.bz2".format(fn)
    if os.path.exists(zf):
        say("reading {}".format(zf))
        return decompress_pickle(zf)
    #
    if os.path.exists(fn):
        with open(fn, 'rb') as filehandle:
            data = pickle.load(filehandle)
        say("building {}".format(zf))
        compress_pickle(data, zf)
        return data
    error("pickle file {}?".format(fn))
    return None

# variables
# dataset directory
dd = os.environ['HOME'] + '/recerca/connexio-guifinet/meshmon/py-nade/ml/datasets/'

ftest = "ml-rt-testing_dataset.pkl"
testing_mat = load_and_compress_pickle(dd + ftest)

ftrain = "ml-rt-training_dataset.pkl"
training_mat = load_and_compress_pickle(dd + ftrain)

# anomaly interval (gateway outage)
anom = ['2021-04-14 01:55:00', '2021-04-14 18:10:00']

classifiers.update({
    'Cluster-based Local Outlier Factor (CBLOF)':
        CBLOF(contamination=contamination_est,
              check_estimator=False,
              n_clusters=8,
              alpha=0.9,
              beta=5,
              use_weights=False,
              random_state=random_state,
              n_jobs=-1),
    'Isolation Forest': IForest(n_estimators=20,
                                max_samples=0.7,
                                contamination=contamination_est,
                                random_state=random_state,
                                max_features=1.0,
                                bootstrap=False,
                                n_jobs=-1),
})

#
# functions
#
def error(msg):
    if not type(msg) is str:
        msg = str(msg)
    click.secho(msg, fg="red")

def get_metrics_from_node(data, id):
    """Returns a DF with the routing metrics of node id along the metrics
    matrices stored in data. Each matrix in data['metric'] corresponds to
    matrices of a capture with data['date'] in argument date. The index of the
    DF contains the date and each colum the metric to a node with data['uid']
    as column name.
    """
    nrows = len(data['metric']) # number of captures
    ncols = data['metric'][0].shape[0] # number of nodes (dimension of the metrics matrices)
    metrics = np.empty((nrows, ncols))
    for i in range(nrows):
        metrics[i] = data['metric'][i][id] # metrics of node id to reach every
                                 # other node in capture i
    metrics[:,id] = 0 # set self maximum metric to 0
    df_metrics = pd.DataFrame(metrics, columns=data['uid'])
    df_metrics.index = pd.to_datetime(data['date'])
    return df_metrics

def read_data_file(fname, force, callf, args):
    """ Reads data from an existing file or calls callf to create it
    otherwise. 
    """
    if not force and (os.path.isfile(fname) or os.path.isfile(fname+'.bz2')):
        say("reading file: " + fname)
        if 'VAE' in str(fname):
            res = load_and_compress_pickle(fname)
            res['clf'].model_ = res['clf']._build_model()
        else:
            res = load_and_compress_pickle(fname)
    else:
        error("building file: " + fname)
        res = callf(**args)
        if res != None: # and not 'VAE' in str(fname):
            if 'VAE' in str(fname):
                model = res['clf'].model_
                res['clf'].model_ = None
                compress_pickle(res, fname)
                res['clf'].model_= model
            else:
                compress_pickle(res, fname)
    return res

def build_model(clf_name, clf, df, df_testing):
    """ 
    """
    start_training_time = time.time()
    print('fitting', clf_name)
    clf.fit(df)
    stop_training_time = time.time() - start_training_time
    print("Completed {} training in:  {}".format(clf_name, stop_training_time))
    scores_pred = clf.decision_function(df) * -1
    scores_pred_test = clf.decision_function(df_testing) * -1
    # predictions_training[f'{clf_name} df'] = scores_pred
    start_prediciton_time = time.time()
    y_pred_n = clf.predict(df)
    y_pred_idx = df.index[y_pred_n==1]
    y_pred_test = clf.predict(df_testing)
    y_pred_idx_test = df_testing.index[y_pred_test==1]
    stop_prediction_time = time.time() - start_prediciton_time
    print("Completed {} prediction in:  {}".format(clf_name, stop_prediction_time))
    print("Training decision function ...")
    return {'stop_training_time': stop_training_time,
            'stop_prediction_time': stop_prediction_time, 
            'clf_name': clf_name, 'clf': clf,
            'y_pred_idx': y_pred_idx,
            'y_pred_idx_test': y_pred_idx_test,
            'scores_pred': scores_pred,
            'scores_pred_test': scores_pred_test}

def count_anom(dates, interval):
    return sum([(np.datetime64(f) > np.datetime64(interval[0])) &
                (np.datetime64(f) < np.datetime64(interval[1]))
                for f in dates])

def id2uid(id, uid):
    """ uid: dictionary {uid: id} """
    return list(uid.values()).index(id)

# compute anomalies for clf_name
def compute_method(training_mat, testing_mat, clf_name, id, fid=None):
    """
    anomaly detection for node id using the training_mat,testing_mat datasets.
    """
    global classifiers
    metrics_training = get_metrics_from_node(training_mat, id)
    metrics_testing = get_metrics_from_node(testing_mat, id)
    # take anomalies only on dates where the node is alive
    valid_days = [sum(metrics_testing.iloc[id]>0)>0 for id in range(metrics_testing.shape[0])]
    if sum(valid_days) == 0:
        error("skipping node {} ({}): not alive during any testing captures".format(
            id, uid.uid2hname(id2uid(id, training_mat['uid']))))
        return None
    metrics_testing = metrics_testing.iloc[valid_days]
    clf = classifiers[clf_name]
    model_dir = os.getcwd() + "/models"
    filen = os.path.join(model_dir, 
                         "clf-{}-{}-{}.pkl".format(type(clf).__name__, fid, id) if fid 
                         else "clf-{}-{}.pkl".format(type(clf).__name__, id))
    pkl = read_data_file(
        filen, False, build_model, 
        args={'clf_name': clf_name, 'clf': clf,
              'df': metrics_training,
              'df_testing': metrics_testing})
    pkl['stop_training_time']
    pkl['stop_prediction_time']
    return pkl

#
# plotting
#
def plot_anom(anom_count, node_count, method, anom=None, title=None, fname=None):
    """ Plot the anomalies from anom_count and the number of nodes in node_count.
    """
    df = anom_count.copy()
    df[df['count'] == 0] = None # remove points with 0 anomalies from plot
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(8,5))
    df.sort_values(by=['index'], ascending=True).plot.scatter(x='index',y='count', fontsize=10, ax=ax, c='red')
    # node_count.plot.line(x='index', ax=ax)
    node_count.plot(drawstyle="steps", x='index', ax=ax)
    if anom:
        [ax.axvline(np.datetime64(v), color="r", ls='--') for v in anom]
        # [ax.axhline(y, color="b", ls='--') for y in [len(training_mat['uid']), len(training_mat['uid'])/2]]
        ax.set_title("{} {}/{}".format(
            method,
            count_anom(anom_count[df['count']>node_count['vote']]['index'], anom),
            count_anom(testing_mat['date'], anom)))
    if title:
        ax.set_title("{} {}".format(method, title))
    ax.set_xlabel("date (testing set)")
    ax.set_ylabel("Anomaly count")
    plt.xticks(fontsize=10, rotation=45)
    plt.show()
    if fname:
        save_figure(fname)

def count_anom_from_nodes(date, ML_anom):
    """ Returns a DF with dates and the amount of anomalies detected by ML
    in each date.
    """
    res = {d:0 for d in date}
    for f in ML_anom.values():
        for d in f['y_pred_idx_test']:
            if d in res:
                res[d] = res[d] + 1
            else:
                print("date? {}".format(d))
    return pd.DataFrame({'index': list(res.keys())}).join(
        pd.DataFrame({'count': list(res.values())}))

def get_live_node(d, id):
    """
    returns a working node from capture c in d
    """
    for i in range(d[id][0].shape[0]):
        if(d[id][i,i] == 128000000000):
            return i
    error("live node {}?".format(id))
    return 0

def get_node_count(test_mat):
    live_node = [get_live_node(test_mat['metric'], i) for i in range(len(test_mat['date']))]
    # DF with how many nodes are alive in every capture
    return pd.DataFrame.from_dict(
        orient='columns', 
        data={'index': test_mat['date'],
              'nodes':
              [sum(test_mat['metric'][i][live_node[i]]>-1) for i in range(len(test_mat['date']))],
              'vote':
              [sum(test_mat['metric'][i][live_node[i]]>-1)/2 for i in range(len(test_mat['date']))]})

def save_figure(file, font=14):
    plt.rcParams.update({'font.size': font})
    plt.savefig('figures/'+file, format='pdf',
                bbox_inches='tight', pad_inches=0)

def unify_features(train, test):
    """ Supress row/columns not common in train/test
    """
    def get_common_features(f1, f2):
        return list(set.intersection(set(f1), set(f2)))
    #
    def unify_uid(d, common_uid, name):
        def rm_row_col(d, idx):
            d = np.delete(d, idx, axis=0)
            d = np.delete(d, idx, axis=1)
            return d
        #
        uid_dif = set.difference(set(d['uid']), set(common_uid))
        if len(uid_dif) > 0:
            idx_dif = [d['uid'][i] for i in uid_dif]
            say("removing nodes from {} set: {}".format(name, uid_dif))
            d['uid'] = {common_uid[i]:i for i in range(len(common_uid))}
            for id in range(len(d['rt'])):
                for name in ['rt', 'adj', 'metric']:
                    d[name][id] = rm_row_col(d[name][id], idx_dif)
        return d
    common_uid = get_common_features(train['uid'], test['uid'])
    train = unify_uid(train, common_uid, 'training')
    test = unify_uid(test, common_uid, 'testing')
    return train, test

def get_ML_data(ftrain, date, ML):
    """ compute anomalies using training data in "ftrain" and testing data in "ml-rt-date.pkl"
    """
    if ML == 'IF':
        clf_short_name = "IForest"
        clf_name = 'Isolation Forest'
    elif ML == 'CBLOF':
        clf_short_name = "CBLOF"
        clf_name = 'Cluster-based Local Outlier Factor (CBLOF)'
    elif ML == 'VAE':
        clf_short_name = "VAE"
        clf_name = 'Variational auto encoder (VAE)'
    else:
        error("ML {}?".format(ML))
        return None
    model_dir = os.getcwd() + "/models"
    fname = os.path.join(model_dir, "clf-{}-{}.pkl".format(clf_short_name, date))
    if os.path.isfile(fname) or os.path.isfile(fname+'.bz2'):
        return load_and_compress_pickle(fname)
    #
    say("building {}".format(fname))
    say("reading training_mat ({})".format(ftrain))
    training_mat = load_and_compress_pickle(dd + ftrain)
    #
    ftest = "ml-rt-{}.pkl".format(date)
    say("reading testing_mat ({})".format(ftest))
    test_mat = load_and_compress_pickle(dd + ftest)
    #
    train, test_mat = unify_features(training_mat, test_mat)
    ML_anom = {} # dictionary with the anomalies detected by ML for each node
    for id in range(len(train['uid'])):
        pkl_ML = compute_method(train, test_mat, clf_name, id, date)
        if pkl_ML:
            ML_anom.update({id: pkl_ML})
            print("id {} ({}), anomalies: {}".format(
                id, uid.uid2hname(id2uid(id, train['uid'])), 
                len(pkl_ML['y_pred_idx_test'])))
    node_count = get_node_count(test_mat)
    anom_count_ML = count_anom_from_nodes(test_mat['date'], ML_anom)
    compress_pickle((ML_anom, node_count, anom_count_ML), fname)
    return ML_anom, node_count, anom_count_ML

if False: '''
#
# build IF data
#
IF_anom = {} # dictionary with the anomalies detected by IF for each node
for id in range(len(training_mat['uid'])):
    pkl_IF = compute_method(training_mat, testing_mat, 'Isolation Forest', id)
    if pkl_IF:
        IF_anom.update({id: pkl_IF})
        print("id {} ({}), anomalies: {}".format(
        id, uid.uid2hname(id2uid(id, training_mat['uid'])), 
        count_anom(pkl_IF['y_pred_idx_test'], anom)))

#
# build CBLOF data
#
CBLOF_anom = {} # dictionary with the anomalies detected by CBLOF for each node
for id in range(len(training_mat['uid'])):
    pkl_CBLOF = compute_method(training_mat, testing_mat, 'Cluster-based Local Outlier Factor (CBLOF)', id)
    if pkl_CBLOF:
        CBLOF_anom.update({id: pkl_CBLOF})
        print("id {} ({}), anomalies: {}".format(
        id, uid.uid2hname(id2uid(id, training_mat['uid'])), 
        count_anom(pkl_CBLOF['y_pred_idx_test'], anom)))

#
# build VAE data
#
VAE_anom = {} # dictionary with the anomalies detected by VAE for each node
for id in range(len(training_mat['uid'])):
    pkl_VAE = compute_method(training_mat, testing_mat, 'Variational auto encoder (VAE)', id)
    if pkl_VAE:
        VAE_anom.update({id: pkl_VAE})
        print("id {} ({}), anomalies: {}".format(
            id, uid.uid2hname(id2uid(id, training_mat['uid'])), 
            count_anom(pkl_VAE['y_pred_idx_test'], anom)))

#
# plot ressults
#
node_count = get_node_count(testing_mat)

#
# count the anomalies found in IF_anom
#
anom_count_IF = count_anom_from_nodes(testing_mat['date'], IF_anom)
len(anom_count_IF)

plot_anom(anom_count_IF, node_count, 'IF', anom)

# save_figure("anomalies-using-metrics-IF.pdf")

#
# count the anomalies found in CBLOF_anom
#
anom_count_CBLOF = count_anom_from_nodes(testing_mat['date'], CBLOF_anom)
len(anom_count_CBLOF)

plot_anom(anom_count_CBLOF, node_count, 'CBLOF', anom)

# save_figure("anomalies-using-metrics-CBLOF.pdf")

#
# count the anomalies found in VAE_anom
#
anom_count_VAE = count_anom_from_nodes(testing_mat['date'], VAE_anom)
len(anom_count_VAE)

plot_anom(anom_count_VAE, node_count, 'VAE', anom)

# save_figure("anomalies-using-metrics-VAE_anom.pdf")
'''

#
# other months
#

if False: '''
#
# IF
#
date = "21-04"
IF_anom, node_count, anom_count_ML = get_ML_data("ml-rt-training_dataset.pkl", date, 'IF')
plot_anom(anom_count_ML, node_count, 'IF', title=date, fname="ml-IF-{}.pdf".format(date))

# building multiple files
year = '21'
for m in range(3, 4+1):
    date = "{}-{:02d}".format(year, m)
    print(date)
    IF_anom, node_count, anom_count_ML = get_ML_data("ml-rt-training_dataset.pkl", date, 'IF')

# plotting multiple files
year = '21'
for m in range(3, 4+1):
    date = "{}-{:02d}".format(year, m)
    IF_anom, node_count, anom_count_ML = get_ML_data("ml-rt-training_dataset.pkl", date, 'IF')
    plot_anom(anom_count_ML, node_count, 'IF', title=date, fname="ml-IF-{}.pdf".format(date))
'''

#
# CBLOF
#
# date = "21-04"
# CBLOF_anom, node_count, anom_count_ML = get_ML_data("ml-rt-training_dataset.pkl", date, 'CBLOF')
# plot_anom(anom_count_ML, node_count, 'CBLOF', title=date, fname="ml-CBLOF-{}.pdf".format(date))

# building multiple files
year = '21'
for m in range(3, 12+1):
    date = "{}-{:02d}".format(year, m)
    print(date)
    CBLOF_anom, node_count, anom_count_ML = get_ML_data("ml-rt-training_dataset.pkl", date, 'CBLOF')

# plotting multiple files
# year = '21'
# for m in range(3, 4+1):
#     date = "{}-{:02d}".format(year, m)
#     CBLOF_anom, node_count, anom_count_ML = get_ML_data("ml-rt-training_dataset.pkl", date, 'CBLOF')
#     plot_anom(anom_count_ML, node_count, 'CBLOF', title=date, fname="ml-CBLOF-{}.pdf".format(date))

exit
#
# VAE
#
date = "21-04"
VAE_anom, node_count, anom_count_ML = get_ML_data("ml-rt-training_dataset.pkl", date, 'VAE')
plot_anom(anom_count_ML, node_count, 'VAE', title=date, fname="ml-VAE-{}.pdf".format(date))

# building multiple files
year = '21'
for m in range(3, 4+1):
    date = "{}-{:02d}".format(year, m)
    print(date)
    VAE_anom, node_count, anom_count_ML = get_ML_data("ml-rt-training_dataset.pkl", date, 'VAE')

# plotting multiple files
year = '21'
for m in range(3, 4+1):
    date = "{}-{:02d}".format(year, m)
    VAE_anom, node_count, anom_count_ML = get_ML_data("ml-rt-training_dataset.pkl", date, 'VAE')
    plot_anom(anom_count_ML, node_count, 'VAE', title=date, fname="ml-VAE-{}.pdf".format(date))

#
#
#
exit
#
# testing
#
node_count.head(50)
capture = 4
[sum(testing_mat['metric'][capture][i]>-1) for i in range(testing_mat['metric'][capture].shape[0])]
