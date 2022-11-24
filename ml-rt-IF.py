#!/usr/bin/python3 -d
# -*- coding: utf-8 -*-
# process training_dataset.pkl and testing_dataset.pkl with RT datasets
# (c) Llorenç Cerdà-Alabern, September 2022.
# debug: import pdb; pdb.set_trace()
# https://docs.h5py.org/en/stable/quick.html

# imports
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
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as font_manager
plt.ion()  # interactive non-blocking mode

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
if 'tensorflow' in sys.modules:
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.models.vae import VAE
else:
    print("skipping VAE, tensorflow not installed")
import time
import shap
import pickle

# wd
pwd = os.getcwd()
print('pwd: ' + pwd)
if not os.path.exists("ml-rt.pl"):
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

# variables
# dataset directory
dd = os.environ['HOME'] + '/recerca/connexio-guifinet/meshmon/py-nade/ml/datasets/'

with open(dd + "testing_dataset.pkl", 'rb') as filehandle:
    testing_mat = pickle.load(filehandle)

with open(dd + "training_dataset.pkl", 'rb') as filehandle:
    training_mat = pickle.load(filehandle)

# anomaly interval (gateway outage)
anom = ['2021-04-14 01:55:00', '2021-04-14 18:10:00']

# pyod parameters
contamination_est = 0.005
# random_state = np.random.RandomState(42)
random_state  = 42
classifiers = {
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
}

if 'VAE' in sys.modules: 
    classifiers.update({
        'Variational auto encoder (VAE)':
        VAE(encoder_neurons=[128, 64, 32], decoder_neurons=[32, 64, 128], 
            epochs=30, batch_size=32,
            dropout_rate=0.2,contamination=contamination_est, verbose=1)})
else:
    print("skipping VAE classifier, library not installed")

#
# functions
#
def error(msg):
    if not type(msg) is str:
        msg = str(msg)
    click.secho(msg, fg="red")

def say(msg):
    if not type(msg) is str:
        msg = str(msg)
    click.secho(msg, fg="green")

def get_metrics_from_node(data, date, id):
    """Returns a DF with the routing metrics of node id along the metrics
    matrices stored in data. Each matrix in data corresponds to
    matrices of a capture with date in argument date. The index of the
    DF contains the date and each colum the metric to a node with uid
    as column name.
    """
    nrows = len(data) # number of captures
    ncols = data[0].shape[0] # number of nodes (dimension of the metrics matrices)
    metrics = np.empty((nrows, ncols))
    for i in range(nrows):
        metrics[i] = data[i][id] # metrics of node id to reach every
                                 # other node in capture i
    metrics[:,id] = 0 # set self maximum metric to 0
    df_metrics = pd.DataFrame(metrics, columns=training_mat['uid'])
    df_metrics.index = pd.to_datetime(date)
    return df_metrics

def read_data_file(fname, force, callf, args):
    """ Reads data from an existing file or calls callf to create it
    otherwise. 
    """
    if not force and os.path.isfile(fname):
        say("reading file: " + fname)
        with open(fname, 'rb') as filehandle:
            if 'VAE' in str(fname):
                res = pickle.load(filehandle)
                res['clf'].model_ = res['clf']._build_model()
            else:
                res = pickle.load(filehandle)
    else:
        error("building file: " + fname)
        res = callf(**args)
        if res != None: # and not 'VAE' in str(fname):
            with open(fname, 'wb') as filehandle:
                if 'VAE' in str(fname):
                    model = res['clf'].model_
                    res['clf'].model_ = None
                    pickle.dump(res, filehandle)
                    res['clf'].model_= model
                else:
                    pickle.dump(res, filehandle)
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

# IF
IF_anom = {} # dictionary with the anomalies detected by IF for each node
def compute_IF(id, anom):
    """
    anomaly detection for node id using the training_mat,testing_mat datasets.
    """
    global IF_anom
    global training_mat,testing_mat
    metrics_training = get_metrics_from_node(training_mat['metric'], training_mat['date'], id)
    metrics_testing = get_metrics_from_node(testing_mat['metric'], testing_mat['date'], id)
    # take anomalies only on dates where the node is alive
    valid_days = [sum(metrics_testing.iloc[id]>0)>0 for id in range(metrics_testing.shape[0])]
    if sum(valid_days) == 0:
        error("skipping node {} ({}): not alive during any testing captures".format(
            id, uid.uid2hname(id2uid(id, training_mat['uid']))))
        return (0, 0)
    metrics_testing = metrics_testing.iloc[valid_days]
    clf_name = 'Isolation Forest'
    clf = classifiers[clf_name]
    model_dir = os.getcwd() + "/models"
    filen = os.path.join(model_dir, 
                         "clf-{}-{}.pkl".format(type(clf).__name__, id))
    pkl_IF = read_data_file(
        filen, False, build_model, 
        args={'clf_name': clf_name, 'clf': clf,
              'df': metrics_training,
              'df_testing': metrics_testing})
    pkl_IF['stop_training_time']
    pkl_IF['stop_prediction_time']
    IF_anom.update({id: pkl_IF})
    return count_anom(pkl_IF['y_pred_idx_test'], anom) # 10/4

#
# plotting
#
def plot_anom(anom_count, node_count):
    """ Plot the anomalies from anom_count and the number of nodes in node_count.
    """
    df = anom_count.copy()
    df[df['count'] == 0] = None # remove points with 0 anomalies from plot
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(8,5))
    df.sort_values(by=['index'], ascending=True).plot.scatter(x='index',y='count', fontsize=10, ax=ax, c='red')
    node_count.plot.line(x='index', ax=ax)
    [ax.axvline(np.datetime64(v), color="r", ls='--') for v in anom]
    # [ax.axhline(y, color="b", ls='--') for y in [len(training_mat['uid']), len(training_mat['uid'])/2]]
    ax.set_title("{}/{}".format(
        count_anom(anom_count[df['count']>node_count['vote']]['index'], anom),
        count_anom(testing_mat['date'], anom)))
    ax.set_xlabel("date (testing set)")
    ax.set_ylabel("Anomaly count")
    plt.xticks(fontsize=10, rotation=45)
    plt.show()

def count_anom_from_nodes(date, IF_anom):
    """ Returns a DF with dates and the amount of anomalies detected by IF
    in each date.
    """
    res = {d:0 for d in date}
    for f in IF_anom.values():
        for d in f['y_pred_idx_test']:
            if d in res:
                res[d] = res[d] + 1
            else:
                print("date? {}".format(d))
    return pd.DataFrame({'index': list(res.keys())}).join(
        pd.DataFrame({'count': list(res.values())}))

def get_live_node(d):
    """
    returns a working node from capture c in d
    """
    for i in range(d[0].shape[0]):
        if(d[i,i] == 128000000000):
            return i
    return None

def save_figure(file, font=14):
    plt.rcParams.update({'font.size': font})
    plt.savefig('figures/'+file, format='pdf',
                bbox_inches='tight', pad_inches=0)

#
# build IF data
#
for id in range(len(training_mat['uid'])):
    print("id {} ({}), anomalies: {}".format(
        id, uid.uid2hname(id2uid(id, training_mat['uid'])), compute_IF(id, anom)))

#
# plot ressults
#

live_node = [get_live_node(testing_mat['metric'][i]) for i in range(len(testing_mat['date']))]
# DF with how many nodes are alive in every capture
node_count = pd.DataFrame.from_dict(
    orient='columns', 
    data={'index': testing_mat['date'],
          'nodes':
          [sum(testing_mat['metric'][i][live_node[i]]>-1) for i in range(len(testing_mat['date']))],
          'vote':
          [sum(testing_mat['metric'][i][live_node[i]]>-1)/2 for i in range(len(testing_mat['date']))]})

# count the anomalies found in IF_anom
anom_count = count_anom_from_nodes(testing_mat['date'], IF_anom)
len(anom_count)

plot_anom(anom_count, node_count)

# save_figure("anomalies-using-metrics-IF.pdf")
