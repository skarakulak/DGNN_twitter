# DGNN on Twitter Hate-Speech Data

[Original Paper](https://arxiv.org/abs/1803.08977), [Data](https://www.kaggle.com/manoelribeiro/hateful-users-on-twitter#users_clean.graphml), [Repo](https://github.com/manoelhortaribeiro/GraphSageHatefulUsers), [Blog Post](https://medium.com/stellargraph/can-graph-machine-learning-identify-hate-speech-in-online-social-networks-58e3b80c9f7e), [Tweets Data](https://www.dropbox.com/sh/ayt6wcjzczhhtwp/AADS7aDFIiIbh-HtCaxdwsHqa?dl=0)

![twitter graph](imgs/twitter_graph.png)



## Data Analysis

- How much they retweet?

  ![rt0](imgs/rt_a_1.png)

  ![rt0](imgs/rt_a_2.png)

  ![rt0](imgs/rt_a_3.png)

- How many times they got retweeted?

  ​		

  ![rt0](imgs/rt_b_1.png)

  ![rt0](imgs/rt_b_2.png)

  ![rt0](imgs/rt_b_3.png)



----------

- How much they reply?

  ![rt0](imgs/rp_a_1.png)

  ![rt0](imgs/rp_a_2.png)

  ![rt0](imgs/rp_a_3.png)

- How many times they receive a reply?

  ​		

  ![rt0](imgs/rp_b_1.png)

  ![rt0](imgs/rp_b_2.png)

  ![rt0](imgs/rp_b_3.png)

Additional remarks from paper about node features:

> We define “hateful user” and “hate speech” according to Twitter’s guidelines. For the purposes of this paper, “hate speech” is any type of content that ‘promotes violence against or directly attack or threaten other people on the basis of race, ethnicity, national origin, sexual orientation, gender, gender identity, religious affiliation, age, disability, or disease.” (Twitter 2017) On the other hand, “hateful user” is a user that, according to annotators, endorses such type of content  
> hateful users tweet more frequently, follow more people each day and their accounts are more short-lived and recent  
> Although hateful users have less followers, the median for several network centrality measures in the retweet network is higher for these users.  
> Hateful users do not seem to behave like spammers.  
> Choice of vocabulary is different: words related to hate, anger and politics occur less often when compared to their normal counterparts, and words related to masculinity, love and curses occur more often  
> Hateful users are 71 times more likely to retweet other hateful users and suspended users are 11 times more likely to retweet other suspended users  
> Hateful users are “power users” in the sense that they tweet more, in shorter intervals, favorite more tweets by other people and follow other users more  


### Remarks on the Dataset

Data Collection

> We represent the connections among users in Twitter using the retweet network (Cha et al. 2010). Sampling the retweet network is hard as we can only observe out-coming edges (due to API limitations), and as it is known that any unbiased in-degree estimation is impossible without sampling most of these “hidden” edges in the graph (Ribeiro et al. 2012). Acknowledging this limitation, we employ Ribeiro et al. Direct Unbiased Random Walk algorithm, which estimates out-degrees distribution efficiently by performing random jumps in an undirected graph it constructs online (Ribeiro, Wang, and Towsley 2010). Fortunately, in the retweet graph the outcoming edges of each user represent the other users she - usually (Guerra et al. 2017) - endorses. With this strategy, we collect a sample of Twitter retweet graph with 100,386 users and 2,286,592 retweet edges along with the 200 most recent tweets for each users, as shown in Figure 1. This graph is unbiased w.r.t. the out degree distribution of nodes.

How they selected which nodes to label:

> As the sampled graph is too large to be annotated entirely, we need to select a subsample to be annotated. If we choose tweets uniformly at random, we risk having a very insignificant percentage of hate speech in the subsample. On the other hand, if we choose only tweets that use obvious hate speech features, such as offensive racial slurs, we will stumble in the same problems pointed in previous work. We propose a method between these two extremes. We:
>
> - Create a lexicon of words that are mostly used in the context of hate speech. This is unlike other work (Davidson et al. 2017) as we do not consider words that are employed in a hateful context but often used in other contexts in a harmless way (e.g. '*n\*gger*'); We use 23 words such as '*holohoax*', '*racial treason*' and '*white genocide*', handpicked from [Hatebase.org](hatebase.org) (Hate Base 2017), and ADL’s hate symbol database (ADL 2017). 
>
> - Run a diffusion process on the graph based on DeGroot’s Learning Model (Golub and Jackson 2010), assigning an initial belief $p_i^0=1$ to each user $u_i$ who employed the words in the lexicon; This prevents our sample from being excessively small or biased towards some vocabulary. 
>
>   ![datacol_diffusion](imgs/datacol_diffusion.png)
>
> - Divide the users in 4 strata according to their associated beliefs after the diffusion process, and perform a stratified sampling, obtaining up to 1500 user per strata.

How did annotaters label:

> Annotators were asked to consider the entire profile (limiting the tweets to the ones collected) rather than individual publications or isolate words and were given examples of terms and codewords in ADL’s hate symbol database. Each user profile was independently annotated by 3 annotators, and, if there was disagreement, up to 5 annotators.



### Work on the Logistic Model for Baseline

#### Data

Out of 1037 features:

- 204 features that are based on a user's attributes and tweet lexicon
- 300 glove, and 300 c_glove features
- Neighborhood aggregated same features ‘c_’Sentiment, hashtags, centrality measures like 'betweenness', 'eigenvector', 'in_degree', 'out_degree'
- Sentiment, hashtags, centrality measures like 'betweenness', 'eigenvector', 'in_degree', 'out_degree'

Only 204 features mentioned below are used

> ['statuses_count', 'followers_count', 'followees_count',……... 
>
> 'negotiate_empath', 'vehicle_empath', 'science_empath', 'timidity_empath', 'gain_empath', 'swearing_terms_empath', 'office_empath', 'tourism_empath', 'computer_empath',……..... 
>
> 'subjectivity', 'number hashtags', 'tweet number', 'retweet number', 'quote number', 'status length', 'number urls', 'baddies', 'mentions']

Train/test split:
60% train, 40% test
-> Data may be small for learning deep models

#### Results for 204 feature baselines

![baseline_1](imgs/baseline2.png)

**Logistic regression** Auc: 0.87, Accuracy: 0.84
**MLP** *(2 hidden layers, 64 and 128 dim)* Auc: 0.89



### Citations of the original paper

- Network Representation Learning from Alibaba [[1]](https://www.ijcai.org/proceedings/2018/0438.pdf) which propose an encoder decoder network for n-gram task of predicting neighbors features, and then using the user encodings for link prediction and node classification
- Survey on Social Media-based User Embedding[[2]](https://arxiv.org/abs/1907.00725) 
- NLP approaches with tweet classification [[3]](https://www.aclweb.org/anthology/P19-1163.pdf)[[4]](https://www.aclweb.org/anthology/W18-4422.pdf)[[5]](https://link.springer.com/article/10.1007/s13278-019-0587-5) and dialogue modeling [[6]](https://arxiv.org/pdf/1909.01362.pdf)
- Outlier detection on retweets using features from source, target, and tweet[[7]](https://link.springer.com/chapter/10.1007/978-3-030-14459-3_12)[[8]](https://arxiv.org/abs/1906.02541)[[9]](https://www-complexnetworks.lip6.fr/~lamarche/pdf/botterman_et_al_2019_ODYCCEUS.pdf) for thing like event detection
- Hate detection in other data modalities like images on Instagram[[10]](http://precog.iiitd.edu.in/pubs/Maybe_Look_Closer-IEEE_BigMM.pdf) 
- Survey [[11]](https://link.springer.com/article/10.1007/s00778-019-00569-6) on data management
- Least Recent Influencer Model [[12]](https://dl.acm.org/citation.cfm?id=3326034) to calculate a hand-crafted measure and diffuse it over the network using a similar dataset
- Feature selection [[13]](https://repositorio-aberto.up.pt/bitstream/10216/119511/2/326963.pdf)
- Multimodal Embedding learning for users, tweets and pictures for retrieval tasks[[id]](https://arxiv.org/abs/1905.07075)
- Social studies [[14]](https://ora.ox.ac.uk/objects/uuid:3c32d29d-e2e4-4913-abf8-2a28886f55a7)[[15]](https://wvvw.aaai.org/ojs/index.php/ICWSM/article/view/3354)
- Causal inference on graphs [[16]](https://why19.causalai.net/papers/elena-zheleva.pdf)
- Analyses[[17]](https://arxiv.org/pdf/1901.09735.pdf)[[18]](https://arxiv.org/pdf/1909.10966.pdf)



# TO-DO List

##  Nov 12

- [x] Figure out the design of the dataset and understand the content

  - We have two zip files. One of them contains the data that is publicized on Kaggle. This doesn't have tweets. It has userIDs and edges with timestamps 

  - Non-public data has tweets. IDs are different. The author gave the following warning:

    > **tweets.csv**: The tweets :)! The columns of this file are described on description_tweets.txt
    >
    > **users_neighborhood.csv**: this file has two rows related to the IDs: `user_id` is the id used in the graph which is on Kaggle and `user_id_original` links with the tweets!

- [x] Investigate the statistics of the data. Lei mentioned GNNs might not work if within-the-same-class edges are not more then out-of-the-same-class edges. So check if hateful users are connected more with other hateful users. Stuff about the smoothing.
- [x] Set bag-of-words baselines
- [x] Review the methods that are applied on this dataset. Check the papers that cites the [original paper](https://arxiv.org/abs/1803.08977). 



### Understanding Dataset

- [x] Check if RT's have text

  - Yes

- [x] How they labeled hateful users. Did they classify using the author's tweets, or were retweets considered as well.

  - I have added for this above 

- [x] Are there replies without tweet IDs, directly to the user? (like mentions)

  - Yes. 275K out of 4M
  
- [x] date distribution

  | quantile   | date   |
  | ---- | ---- |
  | 0.00 |  2006-12-17 |
  | 0.05 |  2017-03-04 |
  | 0.10 |  2017-06-24 |
  | 0.15 |  2017-08-11 |
  | 0.20 |  2017-09-05 |
  | 0.25 |  2017-09-20 |
  | 0.30 |  2017-09-29 |
  | 0.35 |  2017-10-06 |
  | 0.40 |  2017-10-11 |
  | 0.45 |  2017-10-15 |
  | 0.50 |  2017-10-18 |
  | 0.55 |  2017-10-20 |
  | 0.60 |  2017-10-23 |
  | 0.65 |  2017-10-24 |
  | 0.70 |  2017-10-25 |
  | 0.75 |  2017-10-26 |
  | 0.80 |  2017-10-27 |
  | 0.85 |  2017-10-28 |
  | 0.90 |  2017-10-29 |
  | 0.95 |  2017-10-30 |
  | 1.00 |  2017-11-01 |
  



### Nov 15

- [x] Read GraphSAGE, review the code of the original paper.
- [ ] Reproduce GraphSAGE results and work on GCN results
- [x] Formulate alternatives for how to make use of *fake-nodes* to add dynamic structure into static graphs.
- [x] Review the previous dynamic implementation and check how to adapt that to current dataset



### Nov 21

- [ ] Get the original code to work. 
- [ ] Make modifications that are necessary to use fake nodes for the original implementation
- [ ] Read [Training on Giant Graphs](https://docs.dgl.ai/tutorials/models/index.html#training-on-giant-graphs) section of the DGL library and the [Streaming Graph Neural Network](https://arxiv.org/abs/1810.10627) paper
- [ ] Start implementing *sampling* on the dynamic GNN and discuss issues
