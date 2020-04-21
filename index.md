### Introduction

Sentiment analysis is the application of natural language processing (NLP) to learn and determine the emotion or feeling associated with a statement. This feeling can be categorized as positive, neutral or negative; sad, angry, or happy; as a rating from 1 to 10; etc.

Our team performs supervised learning with a convolutional neural network model to perform sentiment analysis on movie reviews. The corpus consists of reviews from multiple movie review aggregators such as IMDB and Rotten Tomatoes with corresponding sentiment labels, as found in existing datasets [5][6].

### Methods

The input data consists of labelled plaintext movie reviews from multiple movie review aggregators such as IMDB and Rotten Tomatoes. We use a word embedding tool such as word2vec to map the plaintext words contained in each review to vectors, which are merged to form a feature set representation. Our model categorizes reviews as positive, neutral, or negative, outputted as 3-element vectors ([1, 0, 0], [0, 1, 0], and [0, 0,1], respectively) from a linear layer. Based on the loss between these outputs and the correct sentiment labels, the model is optimized to predict correctly.

The sentiment analysis model may use a convolutional neural network (CNN). Previous work has shown that deep learning [2][4] and CNN’s [1] can be successfully applied to natural language processing tasks. The final layer of the network is a softmax layer with three neurons, with the output of each corresponding to the probability that the given input review is positive, neutral, or negative.

<p align="center"><img src="/assets/architecture.png" alt="Model Architecture" height="800"/></p>

### Potential Results

We will evaluate the accuracy of our model by comparing the model’s predicted rating for each movie review aggregator to the actual rating. We hope to achieve an accuracy of at least 70%, which approaches the results found in existing literature (mid 70% range with classical ML methods such as SVMs [3], above 80% with deep learning [2]).

### Results

<p align="center"><img src="/assets/loss_graph.png" alt="Loss Graph"/></p>

<p align="center"><img src="/assets/accuracy_graph.png" alt="Accuracy Graph"/></p>

We achieved an accuracy of 70.658684% on the test set.

### Discussion

There are inherent problems with our approach. For example, individual reviews may be rated from zero to five stars, but we will classify each review only as positive, neutral, or negative. Hypothetically, if every review corresponded to four stars, our classification would cause our model to consider every review positive and predict a rating of five stars. Realistically, however, this problem should not be that significant since the actual reviews should be distributed over a wider range.

### References

[1] Conneau, Alexis et al. “Very Deep Convolutional Networks for Natural Language Processing.” ArXiv abs/1606.01781 (2016): n. Pag.

[2] “Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.” Deeply Moving: Deep Learning for Sentiment Analysis, nlp.stanford.edu/sentiment/index.html.

[3] Parkhe, V., Biswas, B. Sentiment analysis of movie reviews: finding most important movie aspects using driving factors. Soft Comput 20, 3373–3379 (2016). https://doi.org/10.1007/s00500-015-1779-1

[4] Wu, Jean Y.. “Predicting Sentiment from Rotten Tomatoes Movie Reviews.” (2012).

[5] Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

[6] “Movie Review Data.” Data, www.cs.cornell.edu/people/pabo/movie-review-data/.
