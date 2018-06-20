import pandas as pd
import numpy as np
import time
import tensorflow as tf
from utils import *
from IPNN import IPNN




data = pd.read_table('../data/train.txt',sep=' ')
data = data.drop_duplicates(['instance_id'])
data['context_timestamp'] = data['context_timestamp'].apply(lambda x:
        time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(x)))
data['time_day'] = data['context_timestamp'].apply(lambda x: x.split(' ')[0].split('-')[2])
data['time_half_hour'] = data['context_timestamp'].apply(lambda x: int(x.split(' ')[1].split(':')[0])*2 + int(x.split(' ')[1].split(':')[1])/30)

feature = data.copy()

feature.rename(columns={'instance_id':'id'},inplace=True)
feature.rename(columns={'is_trade':'target'},inplace=True)
feature_train = feature[(feature.time_day < '24') & (feature.time_day > '19')]
feature_test = feature[feature.time_day == '24']
feature_train.drop(['time_day','context_timestamp'],axis=1,inplace=True)
feature_test.drop(['time_day','context_timestamp'],axis=1,inplace=True)


numeric_cols=['shop_review_positive_rate',
        'shop_score_service','shop_score_delivery','shop_score_description']
ignore_cols=['context_id','user_id','item_property_list','predict_category_property','item_id']

fd = FeatureDictionary(dfTrain=feature_train,dfTest=feature_test,numeric_cols=numeric_cols,ignore_cols=ignore_cols)
fd.gen_feat_dict()
dp = DataParser(fd)
train_xi,train_xv,train_y = dp.parse(df=feature_train,has_label=True)
test_xi,test_xv,test_y = dp.parse(df=feature_test,has_label=True)


# params
ipnn_params = {
    "D1":32,
    "embedding_size": 8,
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 1024,
    "learning_rate": 0.001,
}


ipnn_params["feature_size"] = fd.feat_dim
ipnn_params["field_size"] = len(train_xi[0])
ipnn = IPNN(**ipnn_params)

ipnn.fit(train_xi,train_xv,train_y,test_xi,test_xv,test_y)
