import re
import logging
from transformers import *
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, TensorDataset, SequentialSampler, WeightedRandomSampler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

MODELS = [
    (BertModel, BertTokenizer, 'bert-base-uncased'),
    (OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
    (GPT2Model, GPT2Tokenizer, 'gpt2'),
    (CTRLModel, CTRLTokenizer, 'ctrl'),
    (TransfoXLModel, TransfoXLTokenizer, 'transfo-xl-wt103'),
    (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
    (XLMModel, XLMTokenizer, 'xlm-mlm-enfr-1024'),
    (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
    (RobertaModel, RobertaTokenizer, 'roberta-base')
]


class Categories:
    """
    generic function to enumerate categories
    """
    def __init__(self, categs):
        self.categs = categs
        for i, c in enumerate(self.categs):
            self.__dict__[c] = i

    def i2c(self, ind):
        return self.categs[ind]

    def c2i(self, categ):
        return self.__dict__[categ]

    def __str__(self):
        return ', '.join([f'{i}: {c}' for i, c in enumerate(self.categs)])

    def __repr__(self):
        return 'categories: {' + ', '.join([f'{i}: {c}' for i, c in enumerate(self.categs)]) + '}'


class Mapping:
    """
    generic function to handle conversion between IDs and indices
    """
    def __init__(self, l_IDs):
        self.length = len(l_IDs)
        self.IDs = l_IDs
        self.idx = defaultdict(lambda: -1, {v: k for k, v in enumerate(l_IDs)})
        self.IDs.append(-1)

    def ind2ID(self, ind):
        return self.IDs[ind]

    def ID2ind(self, ID):
        return self.idx[ID]

    def idx2IDs(self, l_idx):
        return [self.ind2ID(i) for i in l_idx]

    def IDs2idx(self, l_IDs):
        return [self.ID2ind(ID) for ID in l_IDs]

    def __len__(self):
        return self.length


def preprocess_tweet(text):
    # remove URLs
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)
    text = re.sub(r'http\S+', '', text)
    # remove usernames
    text = re.sub('@[^\s]+', '', text)
    # remove the # in #hashtag
    text = re.sub(r'#([^\s]+)', r'\1', text)
    # XML character references
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', "\"", text)
    text = re.sub(r'&apos;', "'", text)
    return text.strip()


class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def read_examples(input_df):
    """Read a list of `InputExample`s from a dataframe."""
    examples = []
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_input_examples(row):
    return InputExample(unique_id=row[0], text_a=row[1], text_b=None)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_examples_to_features(examples, seq_length, tokenizer):
    """
    Loads a data file into a list of `InputFeature`s.

    https://github.com/dnanhkhoa/pytorch-pretrained-BERT/blob/master/examples/extract_features.py
    """
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


class Tweet:
    def __init__(
            self, tID, uID=None, bert_ft=None,
            t_replies=defaultdict(set), t_quotes=defaultdict(set), t_retweets=defaultdict(set),
            reply_to_user=None, reply_to_tweet=None, quote_to=None, retweet_to=None,
            is_reply=None, is_retweet=None, is_quote=None,
            empty=False, dt_creation=None
    ):
        self.tID = tID
        self.uID = uID
        self.bert_ft = bert_ft
        self.is_reply = is_reply
        self.is_retweet = is_retweet
        self.is_quote = is_quote
        self.reply_to_tweet = reply_to_tweet
        self.reply_to_user = reply_to_user
        self.quote_to = quote_to
        self.retweet_to = retweet_to
        self.empty = empty
        self.dt_creation = dt_creation

    def __repr__(self):
        return '{' + '\n'.join([
            f'{k}: ' +
            ('\n' if isinstance(v, torch.Tensor) else '') +
            f'{v}' for k, v in self.__dict__.items() if k != 'updates']) + '}'

    def _get_user(self, uID, user_dict):
        u = user_dict.get(uID)
        if u is None:
            u = User(uID=uID)
            user_dict[uID] = u
        return u

    def _upd_user_dts(self, uID, tID, linked_tID, linked_uID, tweet_type, user_dict):
        u = self._get_user(uID, user_dict)
        u.updates[0].append(self.dt_creation)
        u.updates[1].append(tweet_type)
        u.updates[2].append(tID)
        u.updates[3].append(linked_tID)
        u.updates[4].append(linked_uID)

    def set_attributes(self, row, tweet_dict=None, user_dict=None):
        self.uID = row.user_id
        self.is_reply = row.rp_flag
        self.is_quote = row.qt_flag
        self.is_retweet = row.rt_flag
        self.dt_creation = row.tweet_creation

        if self.is_reply:
            self.reply_to_user = row.rp_user
            self.reply_to_tweet = row.rp_status
        if self.is_quote:
            self.quote_to = row.qt_status_id
        if self.is_retweet:
            self.retweet_to = row.rt_status_id

    def prepare_upd_table(self, user_dict, tweet_dict, tweet_categs):
        if (not self.is_reply) and (not self.is_quote) and (not self.is_retweet):
            self._upd_user_dts(self.uID, self.tID, -1, -1, tweet_categs.own_tweet, user_dict)
        if self.is_reply:
            self._upd_user_dts(self.reply_to_user, self.reply_to_tweet, self.tID, self.uID, tweet_categs.replied_by,
                               user_dict)
            self._upd_user_dts(self.uID, self.tID, self.reply_to_tweet, self.reply_to_user, tweet_categs.replied,
                               user_dict)
        if self.is_quote:
            linked_tw = tweet_dict.get(self.quote_to)
            if linked_tw is not None:
                linked_uID = linked_tw.uID
                self._upd_user_dts(linked_uID, self.quote_to, self.tID, self.uID, tweet_categs.quoted_by, user_dict)
                self._upd_user_dts(self.uID, self.tID, self.quote_to, linked_uID, tweet_categs.quoted, user_dict)
        if self.is_retweet:
            linked_tw = tweet_dict.get(self.retweet_to)
            if linked_tw is not None:
                linked_uID = linked_tw.uID
                self._upd_user_dts(linked_uID, self.retweet_to, self.tID, self.uID, tweet_categs.retweeted_by,
                                   user_dict)
                self._upd_user_dts(self.uID, self.tID, self.retweet_to, linked_uID, tweet_categs.retweeted, user_dict)


class User:
    def __init__(self, uID, label=None, features=None):
        self.uID = uID
        self.label = label
        self.features = features
        self.updates = [[], [], [], [], []]
        self.df_updates = None
        self.timesteps = None
        self.tweetvecs = None

    def __repr__(self):
        return '{' + '\n'.join([
            f'{k}: {v}' for k, v in self.__dict__.items() if k != 'updates']) + '}'

    def init_df_updates(self):
        self.df_updates = pd.DataFrame({
            'date': pd.Series(self.updates[0], dtype='datetime64[ns]'),
            'tweet_type': pd.Series(self.updates[1], dtype=np.int64),
            'tID': pd.Series(self.updates[2], dtype=np.int64),
            'linked_tID': pd.Series(self.updates[3], dtype=np.int64),
            'linked_uID': pd.Series(
                list(map(lambda x: -1 if x == None else x, self.updates[4])), dtype=np.int64),
        })


class TwitterPratrainingDataset(Dataset):
    def __init__(self, df_labeled, user_dict, tweet_dict, tweet_categs, num_tweets_range=(16, 32), dim_user_feats=522,
                 dim_tweet_feats=768):
        super().__init__()
        self.df_labeled = df_labeled
        self.user_dict = user_dict
        self.tweet_dict = tweet_dict
        self.tweet_categs = tweet_categs
        self.num_tweets_range = num_tweets_range
        self.i2uID = df_labeled.user_id.values
        self.no_result = torch.zeros(1, dim_tweet_feats + 2, dtype=torch.float32)
        self.empty_tweet = torch.zeros(dim_tweet_feats + 2)
        self.empty_tweet[:2] = 1.

        self.user_no_result = torch.zeros(1, dim_user_feats + 2, dtype=torch.float32)
        self.empty_user = torch.zeros(dim_user_feats + 2)
        self.empty_user[:2] = 1.
        self.dim_user_feats = dim_user_feats
        self.dim_tweet_feats = dim_tweet_feats

    def __len__(self):
        return self.df_labeled.shape[0]

    def __getitem__(self, idx):
        uID = self.i2uID[idx]
        u = user_dict[uID]
        user_feats = torch.tensor(u.features)
        y = u.label
        num_tweets = np.random.randint(*self.num_tweets_range)

        sampled_tweets = u.df_updates.sample(n=num_tweets, replace=True, axis=0)

        tweets_own = self.get_features_from_updates_df(
            sampled_tweets[sampled_tweets.tweet_type == tweet_categs.own_tweet],
            user_dict, tweet_dict, get_tID_only=True)
        t_rt_by, linked_t_rt_by = self.get_features_from_updates_df(
            sampled_tweets[sampled_tweets.tweet_type == tweet_categs.retweeted_by],
            user_dict, tweet_dict)
        t_rp_by, linked_t_rp_by = self.get_features_from_updates_df(
            sampled_tweets[sampled_tweets.tweet_type == tweet_categs.replied_by],
            user_dict, tweet_dict)
        t_qt_by, linked_t_qt_by = self.get_features_from_updates_df(
            sampled_tweets[sampled_tweets.tweet_type == tweet_categs.quoted_by],
            user_dict, tweet_dict)
        t_rt, linked_t_rt = self.get_features_from_updates_df(
            sampled_tweets[sampled_tweets.tweet_type == tweet_categs.retweeted],
            user_dict, tweet_dict)
        t_rp, linked_t_rp = self.get_features_from_updates_df(
            sampled_tweets[sampled_tweets.tweet_type == tweet_categs.replied],
            user_dict, tweet_dict)
        t_qt, linked_t_qt = self.get_features_from_updates_df(
            sampled_tweets[sampled_tweets.tweet_type == tweet_categs.quoted],
            user_dict, tweet_dict)

        return ((tweets_own, t_rt_by, t_rp_by, linked_t_rp_by,
                 t_qt_by, linked_t_qt_by, linked_t_rt, t_rp, linked_t_rp,
                 t_qt, linked_t_qt), y)

    #         return (
    #             (user_feats, tweets_own,
    #              t_rt_by, linked_t_rt_by, linked_u_rt_by,
    #              t_rp_by, linked_t_rp_by, linked_u_rp_by,
    #              t_qt_by, linked_t_qt_by, linked_u_qt_by,
    #              t_rt, linked_t_rt, linked_u_rt,
    #              t_rp, linked_t_rp, linked_u_rp,
    #              t_qt, linked_t_qt, linked_u_qt),
    #             y
    #         )

    def replace_none(self, arr, empty_tensor):
        return empty_tensor if arr is None else torch.cat((torch.Tensor([1., 0.]), torch.tensor(arr)))

    def get_features_from_updates_df(self, filtered_df, user_dict, tweet_dict, get_tID_only=False):
        if get_tID_only:
            if filtered_df.shape[0] == 0:
                return self.no_result
            else:
                l_tweet = [tweet_dict.get(k) for k in filtered_df.tID]
                tweets_own = torch.stack(
                    [torch.cat((torch.Tensor([1., 0.]), tw.bert_ft))
                     if tw is not None and tw.bert_ft is not None else self.empty_tweet
                     for tw in l_tweet])
                return tweets_own

        elif filtered_df.shape[0] == 0:
            return self.no_result, self.no_result  # self.user_no_result
        else:
            #             linked_uid = torch.stack(
            #                 [self.replace_none(user_dict[uid].features, self.empty_user)
            #                  if uid > 0 else self.empty_user
            #                  for uid in filtered_df['linked_uID']])
            l_tweets = [tweet_dict.get(k) for k in filtered_df['linked_tID']]
            linked_tid = torch.stack(
                [torch.cat((torch.Tensor([1., 0.]), tw.bert_ft))
                 if tw is not None and tw.bert_ft is not None else self.empty_tweet
                 for tw in l_tweets])

            l_tweet = [tweet_dict.get(k) for k in filtered_df['tID']]
            tid = torch.stack(
                [torch.cat((torch.Tensor([1., 0.]), tw.bert_ft))
                 if tw is not None and tw.bert_ft is not None else self.empty_tweet
                 for tw in l_tweet])

            return tid, linked_tid  # linked_uid


def process_df_updates(v):
    df_upd_g = v.df_updates[
        (v.df_updates.linked_uInd != -1) &
        (v.df_updates.linked_uInd != v.uInd)
        ][['date', 'tweet_type', 'linked_uInd']]
    v_linked_first = df_upd_g[
        ['date', 'linked_uInd']].sort_values(by='date').linked_uInd.drop_duplicates(keep='first').index

    df_upd_merged = pd.merge(
        df_upd_g,
        df_upd_g.loc[v_linked_first][['date', 'linked_uInd']],
        left_on=['date', 'linked_uInd'],
        right_on=['date', 'linked_uInd'],
        how='inner').reset_index(drop=True)

    df_upd_w_counts = pd.concat(
        [df_upd_merged,
         pd.DataFrame(np.zeros((df_upd_merged.shape[0], 6)).astype('int64'), columns=[str(k) for k in range(1, 7)])],
        axis=1
    )

    one_hot = pd.get_dummies(df_upd_w_counts.tweet_type).astype('int64')
    one_hot.columns = [str(k) for k in one_hot.columns]
    df_upd_w_counts.update(one_hot)

    df_upd_final = df_upd_w_counts.drop('tweet_type', axis=1).groupby(['date', 'linked_uInd']).sum().reset_index()
    df_upd_final['uInd'] = np.array([v.uInd] * df_upd_final.shape[0])
    return df_upd_final.set_index('date')


def df_upd_existing_edge(u, temp_df_newedge):
    temp_df = u.df_updates[
        (u.df_updates.linked_uInd != -1) & (u.df_updates.linked_uInd != u.uInd)]
    if (temp_df is None) or (temp_df.shape[0] == 0): return None

    temp_join = pd.merge(
        temp_df,
        temp_df_newedge.loc[[u.uInd]],
        left_on=['date', 'linked_uInd'],
        right_on=['date', 'linked_uInd'],
        how='outer', indicator=True)
    temp_join = temp_join[temp_join._merge == 'left_only'].drop(['tID', 'linked_tID', 'linked_uID', '_merge'], axis=1)
    temp_join = temp_join.reset_index(drop=True)

    if (temp_join is None) or (temp_join.shape[0] == 0): return None
    df_upd_w_counts = pd.concat(
        [temp_join,
         pd.DataFrame(np.zeros((temp_join.shape[0], 6)).astype('int64'), columns=[str(k) for k in range(1, 7)])],
        axis=1
    )
    one_hot = pd.get_dummies(temp_join.tweet_type).astype('int64')
    one_hot.columns = [str(k) for k in one_hot.columns]
    df_upd_w_counts.update(one_hot)

    df_upd_final = df_upd_w_counts.drop('tweet_type', axis=1).groupby(['date', 'linked_uInd']).sum().reset_index()
    df_upd_final['uInd'] = np.array([u.uInd] * df_upd_final.shape[0])
    return df_upd_final.set_index('date')


def calc_norms(g, device='cuda'):
    """ calculate norm of the nodes in the graph and saves it
    with the key 'norm' """
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    norm = norm.to(device)
    g.ndata['norm'] = norm.unsqueeze(1)

#     degs = g.out_degrees().float()
#     norm = torch.pow(degs, -0.5)
#     norm[torch.isinf(norm)] = 0
#     norm = norm.to(device)
#     g.ndata['out_norm'] = norm.unsqueeze(1)

def tstep_default():
    return -1