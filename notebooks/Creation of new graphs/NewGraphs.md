
# Experiments with extended graphs 
These are different graphs which are extensions of the original one using *fake nodes*. It can be tried to use GCN on these graphs, maybe weighing differently different types of edges. I don't use the "c_" attributes but regular ones because those are averaging the embedding with neighbors!

 ## Timestamps
 The total timeframe of *tweets* (not retweets!) is  separated it into $n$ intervals, and created $n$ fake nodes. Then a user tweeting at time $i$ is connected with fake node $i$.  Idea: Information propagation for users with similar twitter patterns (this might be a weaker connection). I originally $n = 100$, but since the amount of tweets over time is increasing, I sort the tweets first and then do $n$ chunks of equal size instead of $n$ intervals of the same length

 ## Clustering of the GloVe Vectors
We cluster the GloVe embedding vectors into $k$ clusters (with k-means for example), where $k$ is a hyperparameter. We then create $k$ fake nodes and connect each user to the fake node associated with its cluster. 
Idea: We want information to be transmitted through similar nodes even though they might be far in the graph (I set originally $k = 50$ so it's "harder" to be similar) 

## Others
These are the two that Joan mentioned, but based on the data available, I came up with other ideas to try, even if they are not strictly time-related (neither is the glove cluster, however), that you can easily create using my code with minor modifications.

- By overall popularity: fake nodes would be intervals of the number of followers
- By clustering of the empathy vectors found in annotated users (although I don't know how meaningful they are compared to the GloVe vectors!)
- By node degree

