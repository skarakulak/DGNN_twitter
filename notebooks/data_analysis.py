# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

# %%
df = pd.read_csv('../twitter_data/nonpublic/users_neighborhood.csv')

# %%
df.head(5)

# %%
u2l = df.set_index('user_id_original').hate
# u2l = u2l[u2l!='other']
# u2l = (u2l == 'hateful').astype(float)

# %%
df.hate

# %%
u2l

# %%
u2l.value_counts()

# %%
df_tweets = pd.read_csv('../twitter_data/nonpublic/tweets.csv',
#         index_col=["date", "loc"], 
        usecols=["user_id", "tweet_id", "rt_flag", "rt_user_id", "rt_status_id"],
        dtype={'user_id':np.int64, 'tweet_id':np.int64, 'rt_flag':np.bool, 'rt_user_id':str, 'rt_status_id':str }
)

df_tweets[['rt_user_id','rt_status_id']] = df_tweets[['rt_user_id','rt_status_id']].fillna("-1").astype(int)

# %%
df_tweets.head()

# %%
rt_tweets = df_tweets[df_tweets.rt_flag == True]

# %%
rt_tweets.head()

# %%
print(f'number of retweets: {rt_tweets.shape[0]}')

# %%
u2u = rt_tweets[['user_id', 'rt_user_id']].drop_duplicates()
print(f'u2u connections: {u2u.shape[0]}')
u2u = u2u.merge(u2l, how='left', left_on='user_id', right_on='user_id_original', suffixes=('', '_tweeting_user'))
u2u = u2u.merge(u2l, how='left', left_on='rt_user_id', right_on='user_id_original', suffixes=('', '_rt_user'))

# %%
plt.figure(figsize=(10,8))
temp = u2u.groupby(['user_id','hate']).size()
temp[temp.index.get_level_values('hate') == 'hateful'].plot.kde(label='hateful', c='r')
temp[temp.index.get_level_values('hate') == 'normal'].plot.kde(label='normal', c='g')
plt.legend(loc='best')
plt.title('Distribution of Retweets Made by a User')
plt.xlabel('number of retweets')
plt.xlim(left=0, right=temp.max())
plt.show()

plt.figure(figsize=(10,8))
temp = u2u[u2u.hate_rt_user=='hateful'].groupby(['user_id','hate']).size()
plt.hist(temp[temp.index.get_level_values('hate') == 'hateful'].values, alpha=0.5, label='Retweeted hateful users', color='r')
temp = u2u[u2u.hate_rt_user=='normal'].groupby(['user_id','hate']).size()
plt.hist(temp[temp.index.get_level_values('hate') == 'hateful'].values, alpha=0.5, label='Retweeted normal users', color='g')
plt.legend(loc='best')
plt.title('How many times hateful users retweet hatefuls and normals')
plt.xlabel('number of retweets')
plt.xlim(left=0, right=temp.max())
plt.show()

plt.figure(figsize=(10,8))
temp = u2u[u2u.hate_rt_user=='hateful'].groupby(['user_id','hate']).size()
plt.hist(temp[temp.index.get_level_values('hate') == 'normal'].values, alpha=0.5, label='Retweeted hateful users', color='r')
temp = u2u[u2u.hate_rt_user=='normal'].groupby(['user_id','hate']).size()
plt.hist(temp[temp.index.get_level_values('hate') == 'normal'].values, alpha=0.5, label='Retweeted  normal users', color='g')
plt.legend(loc='best')
plt.title('How many times normal users retweet hatefuls and normals')
plt.xlabel('number of retweets')
plt.xlim(left=0, right=temp.max())
plt.show()



plt.figure(figsize=(10,8))
temp = u2u.groupby(['rt_user_id','hate_rt_user']).size()
temp[temp.index.get_level_values('hate_rt_user') == 'hateful'].plot.kde(label='hateful', c='r')
temp[temp.index.get_level_values('hate_rt_user') == 'normal'].plot.kde(label='normal', c='g')
plt.legend(loc='best')
plt.title('Distribution of the Number of Times Users Got Retweeted')
plt.xlabel('number of retweets')
plt.xlim(left=0, right=temp.quantile(.975))
plt.show()

plt.figure(figsize=(10,8))
temp = u2u[u2u.hate=='hateful'].groupby(['rt_user_id','hate_rt_user']).size()
plt.hist(temp[temp.index.get_level_values('hate_rt_user') == 'hateful'].values, alpha=0.5, label='by hateful users', color='r')
temp = u2u[u2u.hate=='normal'].groupby(['rt_user_id','hate_rt_user']).size()
plt.hist(temp[temp.index.get_level_values('hate_rt_user') == 'hateful'].values, alpha=0.5, label='by normal users', color='g')
plt.legend(loc='best')
plt.title('How many times hateful users got retweeted')
plt.xlabel('number of retweets')
plt.xlim(left=0, right=temp.max())
plt.show()

plt.figure(figsize=(10,8))
temp = u2u[u2u.hate=='hateful'].groupby(['rt_user_id','hate_rt_user']).size()
plt.hist(temp[temp.index.get_level_values('hate_rt_user') == 'normal'].values, alpha=0.5, label='by hateful users', color='r')
temp = u2u[u2u.hate=='normal'].groupby(['rt_user_id','hate_rt_user']).size()
plt.hist(temp[temp.index.get_level_values('hate_rt_user') == 'normal'].values, alpha=0.5, label='by normal users', color='g')
plt.legend(loc='best')
plt.title('How many times normal users got retweeted')
plt.xlabel('number of retweets')
plt.xlim(left=0, right=temp.max())
plt.show()

# %% [markdown]
# replies

# %%
df_tweets = pd.read_csv('../twitter_data/nonpublic/tweets.csv',
#         index_col=["date", "loc"], 
        usecols=["user_id", "tweet_id", "rp_flag", "rp_user", "rp_status"],
        dtype={'user_id':np.int64, 'tweet_id':np.int64, 'rp_flag':np.bool, 'rp_user':str, 'rp_status':str }
)

df_tweets['rp_user'] = pd.to_numeric(df_tweets['rp_user'], downcast='integer', errors='coerce').fillna(-1)
df_tweets['rp_status'] = pd.to_numeric(df_tweets['rp_status'], downcast='integer', errors='coerce').fillna(-1)

# %%
rp_tweets = df_tweets[df_tweets.rp_flag == True]

# %%
u2u = rp_tweets[['user_id', 'rp_user']].drop_duplicates()
print(f'u2u connections: {u2u.shape[0]}')
u2u = u2u.merge(u2l, how='left', left_on='user_id', right_on='user_id_original', suffixes=('', '_tweeting_user'))
u2u = u2u.merge(u2l, how='left', left_on='rp_user', right_on='user_id_original', suffixes=('', '_rp_user'))

# %%
(rp_tweets.rp_status<0).shape

# %%
plt.figure(figsize=(10,8))
temp = u2u.groupby(['user_id','hate']).size()
temp[temp.index.get_level_values('hate') == 'hateful'].plot.kde(label='hateful', c='r')
temp[temp.index.get_level_values('hate') == 'normal'].plot.kde(label='normal', c='g')
plt.legend(loc='best')
plt.title('Distribution of Replies Given By a User')
plt.xlabel('number of replies')
plt.xlim(left=0, right=temp.max())
plt.show()

plt.figure(figsize=(10,8))
temp = u2u[u2u.hate_rp_user=='hateful'].groupby(['user_id','hate']).size()
plt.hist(temp[temp.index.get_level_values('hate') == 'hateful'].values, alpha=0.5, label='Reply to hateful users', color='r')
temp = u2u[u2u.hate_rp_user=='normal'].groupby(['user_id','hate']).size()
plt.hist(temp[temp.index.get_level_values('hate') == 'hateful'].values, alpha=0.5, label='Reply to normal users', color='g')
plt.legend(loc='best')
plt.title('How many times hateful users reply to hatefuls and normals')
plt.xlabel('number of replies')
plt.xlim(left=0, right=temp.max())
plt.show()

plt.figure(figsize=(10,8))
temp = u2u[u2u.hate_rp_user=='hateful'].groupby(['user_id','hate']).size()
plt.hist(temp[temp.index.get_level_values('hate') == 'normal'].values, alpha=0.5, label='Reply to hateful users', color='r')
temp = u2u[u2u.hate_rp_user=='normal'].groupby(['user_id','hate']).size()
plt.hist(temp[temp.index.get_level_values('hate') == 'normal'].values, alpha=0.5, label='Reply to normal users', color='g')
plt.legend(loc='best')
plt.title('How many times normal users reply to hatefuls and normals')
plt.xlabel('number of replies')
plt.xlim(left=0, right=temp.max())
plt.show()



plt.figure(figsize=(10,8))
temp = u2u.groupby(['rp_user','hate_rp_user']).size()
temp[temp.index.get_level_values('hate_rp_user') == 'hateful'].plot.kde(label='hateful', c='r')
temp[temp.index.get_level_values('hate_rp_user') == 'normal'].plot.kde(label='normal', c='g')
plt.legend(loc='best')
plt.title('Distribution of the Number of Times Users Recieve a Reply')
plt.xlabel('number of replies')
plt.xlim(left=0, right=temp.quantile(.975))
plt.show()

plt.figure(figsize=(10,8))
temp = u2u[u2u.hate=='hateful'].groupby(['rp_user','hate_rp_user']).size()
plt.hist(temp[temp.index.get_level_values('hate_rp_user') == 'hateful'].values, alpha=0.5, label='by hateful users', color='r')
temp = u2u[u2u.hate=='normal'].groupby(['rp_user','hate_rp_user']).size()
plt.hist(temp[temp.index.get_level_values('hate_rp_user') == 'hateful'].values, alpha=0.5, label='by normal users', color='g')
plt.legend(loc='best')
plt.title('How many times hateful users receive a reply')
plt.xlabel('number of replies')
plt.xlim(left=0, right=temp.max())
plt.show()

plt.figure(figsize=(10,8))
temp = u2u[u2u.hate=='hateful'].groupby(['rp_user','hate_rp_user']).size()
plt.hist(temp[temp.index.get_level_values('hate_rp_user') == 'normal'].values, alpha=0.5, label='by hateful users', color='r')
temp = u2u[u2u.hate=='normal'].groupby(['rp_user','hate_rp_user']).size()
plt.hist(temp[temp.index.get_level_values('hate_rp_user') == 'normal'].values, alpha=0.5, label='by normal users', color='g')
plt.legend(loc='best')
plt.title('How many times normal users receive a reply')
plt.xlabel('number of replies')
plt.xlim(left=0, right=temp.max())
plt.show()

# %%

# %%

# %%

# %%
plt.figure(figsize=(10,8))
temp = u2u[u2u.hate_rt_user=='hateful'].groupby(['user_id','hate']).size()
temp[temp.index.get_level_values('hate') == 'hateful'].plot.kde(label='hateful')
temp[temp.index.get_level_values('hate') == 'normal'].plot.kde(label='normal')
# temp[temp.index.get_level_values('hate') == 'other'].plot.kde(label='unlabeled')
plt.legend(loc='best')
plt.title('Distribution of Retweets from Hateful Users')
plt.xlabel('number of retweets')

# %%
# number of edges - quantiles
u2u.groupby(['user_id','hate']).size().groupby(['hate']).quantile([.1,0.25, 0.5, 0.75,.9]).unstack(1)

# %%
u2u.groupby(['user_id','hate']).size().groupby(['hate']).quantile([.1,0.25, 0.5, 0.75,.9]).unstack(1)

# %%
# number of edges - quantiles
u2u[u2u.hate_rt_user=='hateful'].groupby(['user_id','hate']).size().groupby(['hate']).quantile([.1,0.25, 0.5, 0.75,.9]).unstack(1)

# %%
u2u[u2u.hate_rt_user=='normal'].groupby(['user_id','hate']).size().groupby(['hate']).quantile([.1,0.25, 0.5, 0.75,.9]).unstack(1)

# %%
u2u[u2u.hate_rt_user=='other'].groupby(['user_id','hate']).size().groupby(['hate']).quantile([.1,0.25, 0.5, 0.75,.9]).unstack(1)

# %%
u2u[pd.isna(u2u.hate_rt_user)].groupby(['user_id','hate']).size().groupby(['hate']).quantile([.1,0.25, 0.5, 0.75,.9]).unstack(1)

# %% [markdown]
# How much are they retweeted

# %%
# number of edges - quantiles
u2u.groupby(['rt_user_id','hate_rt_user']).size().groupby(['hate_rt_user']).quantile([.1,0.25, 0.5, 0.75,.9]).unstack(1)

# %%
# number of edges - quantiles
u2u[u2u.hate=='normal'].groupby(['rt_user_id','hate_rt_user']).size().groupby(['hate_rt_user']).quantile([.1,0.25, 0.5, 0.75,.9]).unstack(1)

# %%
# number of edges - quantiles
u2u[u2u.hate=='hateful'].groupby(['rt_user_id','hate_rt_user']).size().groupby(['hate_rt_user']).quantile([.1,0.25, 0.5, 0.75,.9]).unstack(1)

# %%
# number of edges - quantiles
u2u[u2u.hate=='other'].groupby(['rt_user_id','hate_rt_user']).size().groupby(['hate_rt_user']).quantile([.1,0.25, 0.5, 0.75,.9]).unstack(1)

# %%

# %%
# number of edges - quantiles
u2u.groupby(['rt_user_id','hate_rt_user']).size().groupby(['hate_rt_user']).quantile([.1,0.25, 0.5, 0.75,.9]).unstack(1)

# %%
u2u.head()

# %%
df_tweets[df_tweets.rt_flag==1].head()

# %%
list(df.columns)

# %%
df.
