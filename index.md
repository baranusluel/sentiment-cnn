### Introduction

Sentiment analysis is the application of natural language processing (NLP) to learn and determine the emotion or feeling associated with a statement. This feeling can be categorized as positive, neutral or negative; sad, angry, or happy; as a rating from 1 to 10; etc.

Our team performs supervised learning with a convolutional neural network model to perform sentiment analysis on movie reviews. The corpus consists of reviews from multiple movie review aggregators such as IMDB and Rotten Tomatoes with corresponding sentiment labels, as found in existing datasets [5][6].

TODO: Background, existing literature. Why we think CNNs could work here

TODO: Motivation, why is this important?

### Dataset and Pre-Processing

TODO: Where we got the dataset, nature of the data

Below are several randomly selected quotes from the movie reviews in the dataset, to demonstrate the nature of the data:

- `hollywood adaptations of plays rarely come off much worse than this b&w one did . there is no kind way to put it otherwise , or to say if only it did this or that the film could have been saved , nor is there reason to say , if the acting was better the film could have been bearable .`
- `wonderland is a rather sugary romance film that is as subtle as a ton of bricks falling on you . you can see its plot developing from a mile away .`
- `director paul verhoeven , whose previous works include movies as diverse as robocop and basic instinct , takes an especially imaginative and fresh approach to science fiction with his new film starship troopers .`

TODO: Visualization, word cloud of the dataset

TODO: Visualize distribution of positive/neutral/negative labels

TODO: What pre-processing we perform, what is word2vec (with visualization)

TODO: Describe training/validation/test data split

### Methods

The input data consists of labelled plaintext movie reviews from multiple movie review aggregators such as IMDB and Rotten Tomatoes. We use a word embedding tool such as word2vec to map the plaintext words contained in each review to vectors, which are merged to form a feature set representation. Our model categorizes reviews as positive, neutral, or negative, outputted as 3-element vectors ([1, 0, 0], [0, 1, 0], and [0, 0,1], respectively) from a linear layer. Based on the loss between these outputs and the correct sentiment labels, the model is optimized to predict correctly.

The sentiment analysis model may use a convolutional neural network (CNN). Previous work has shown that deep learning [2][4] and CNN’s [1] can be successfully applied to natural language processing tasks. The final layer of the network is a softmax layer with three neurons, with the output of each corresponding to the probability that the given input review is positive, neutral, or negative.

TODO: What tools/library we used (Keras)

TODO: Why we picked this model architecture

<p align="center"><img src="./assets/architecture.png" alt="Model Architecture" height="800"/></p>

TODO: What we do against overfitting

TODO: Other implementation details (loss function, early stopping)

### Results

We will evaluate the accuracy of our model by comparing the model’s predicted rating for each movie review aggregator to the actual rating. We hope to achieve an accuracy of at least 70%, which approaches the results found in existing literature (mid 70% range with classical ML methods such as SVMs [3], above 80% with deep learning [2]).

TODO: Learning rate and other hyperparameters

<p align="center"><img src="./assets/loss_graph.png" alt="Loss Graph"/></p>

<p align="center"><img src="./assets/accuracy_graph.png" alt="Accuracy Graph"/></p>

We achieved an accuracy of 70.658684% on the test set.

TODO: Compare to results from literature

### Conclusion

### References

[1] Conneau, Alexis et al. “Very Deep Convolutional Networks for Natural Language Processing.” ArXiv abs/1606.01781 (2016): n. Pag.

[2] “Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.” Deeply Moving: Deep Learning for Sentiment Analysis, nlp.stanford.edu/sentiment/index.html.

[3] Parkhe, V., Biswas, B. Sentiment analysis of movie reviews: finding most important movie aspects using driving factors. Soft Comput 20, 3373–3379 (2016). https://doi.org/10.1007/s00500-015-1779-1

[4] Wu, Jean Y.. “Predicting Sentiment from Rotten Tomatoes Movie Reviews.” (2012).

[5] Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

[6] “Movie Review Data.” Data, www.cs.cornell.edu/people/pabo/movie-review-data/.
