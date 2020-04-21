### Introduction

Sentiment analysis is the application of natural language processing (NLP) to learn and determine the emotion or feeling associated with a statement. This feeling can be categorized as positive, neutral or negative; sad, angry, or happy; as a rating from 1 to 10; etc.

Our team performs supervised learning with a convolutional neural network model to perform sentiment analysis on movie reviews. The corpus consists of reviews from multiple movie review aggregators such as IMDB and Rotten Tomatoes with corresponding sentiment labels, as found in existing datasets [5][6].

TODO: Background, existing literature. Why we think CNNs could work here

#### CNNs

Convolutional Neural Networks (CNNs) are neural networks that use convolutional layers, where each layer essentially applies a sliding window function to some input data. While it is often used in computer vision on images represented as pixel matrices, they can be used in NLP as well, on documents represented as sentence matrices. CNNs use filters that slide over these matrices, and they have been found to perform quite well in extracting useful information such as relationships between words. We chose to develop a CNN model after reading through references on the use of CNNs on sentiment analysis problems. 

TODO: Motivation, why is this important?

### Dataset and Pre-Processing

For our movie review data, we used "Scale Dataset v1.0" by Bo Pang et al [6]. This dataset has been commonly studied in the literature for the task of sentiment analysis [7], which is why we found it appropriate for our investigation.

The dataset contains 5006 total examples, in the form of a tokenized plaintext movie review and an associated label for each example. We have used the 3-class labels provided in the dataset, which are interpreted as 0 (negative), 1 (neutral) and 2 (positive).

Below are several randomly selected quotes from the movie reviews in the dataset, to demonstrate the nature of the data:

- `hollywood adaptations of plays rarely come off much worse than this b&w one did . there is no kind way to put it otherwise , or to say if only it did this or that the film could have been saved , nor is there reason to say , if the acting was better the film could have been bearable .`
- `wonderland is a rather sugary romance film that is as subtle as a ton of bricks falling on you . you can see its plot developing from a mile away .`
- `director paul verhoeven , whose previous works include movies as diverse as robocop and basic instinct , takes an especially imaginative and fresh approach to science fiction with his new film starship troopers .`

TODO: Visualization, word cloud of the dataset

The following bar graph illustrates how the entire dataset is distributed over the possible output labels (negative/neutral/positive). As shown, there are relatively fewer negative examples, so it is worth noting that our results may be affected by a minor class imbalance inherent in the dataset.

<p align="center"><img src="./assets/label_frequencies.png" alt="Frequencies of Labels"/></p>

TODO: What pre-processing we perform, what is word2vec (with visualization)

At the end of our pre-processing, we randomly split the dataset into training, validation and test sets at proportions of 60% / 20% / 20% respectively. This means our training set is composed of 3003 examples.

### Methods

The input data consists of labelled plaintext movie reviews from multiple movie review aggregators such as IMDB and Rotten Tomatoes. We use a word embedding tool such as word2vec to map the plaintext words contained in each review to vectors, which are merged to form a feature set representation. 

#### word2vec
We use the word embedding tool word2vec trained on Google News to map the plaintext words contained in each review to vectors. Word2vec was created using a standard neural network with one hidden layer size of 300. The network was trained on Google News to take in a particular word and predict a probability that every word in the corpus would appear in the surrounding context of the input word. The words are all one-hot encoded (so they are all represented by N x 1 one-hot encoded vector, where N is the number of unique words in the corpus), and the last layer includes the softmax activation function, so each element in the output vector is a probability for the word corresponding to that element's position to be found near the input word. In the sentence "The quick brown fox jumps over the lazy dog", for example, some of the training samples (before one-hot encoding) would be "brown" with the label "fox", "brown" with the label "quick", "fox" with the label "jumps", and so on. This way, when fed a particular word, the network would output the highest probabilities for the words that are most likely to be found near the input word. 

IMAGE

After training, we can extract the weight matrix from the hidden layer, which will be of dimensions 300 x N. If we multiply this matrix and a given word vector (which is of dimension N x 1), we will obtain a 300 x 1 embedding vector corresponding to that word. It makes sense that this vector would embed important contextual information about the input word. Due to the way the network was trained, words that appear in similar contexts will likely have similar word embedding vectors, because the network should produce a similar output context probability vector for words that appear in similar contexts. Thus, in the embedding space, contextually similar words will be clustered together. These vectors capture the semantic meaning of words, which theoretically should lessen the amount of information our predictive model must learn to classify movie review predictions accurately. 

We restrict ourselves to processing the first 500 words of every review. For every review, we stack the corresponding word vectors sequentially, creating a 500 x 300 matrix. If a particular review contains less than 500 words, we pad the rows with zero vectors until the matrix has length 500.  

Our model categorizes reviews as positive, neutral, or negative, outputted as 3-element vectors ([1, 0, 0], [0, 1, 0], and [0, 0,1], respectively) from a linear layer. Based on the loss between these outputs and the correct sentiment labels, the model is optimized to predict correctly.

The sentiment analysis model may use a convolutional neural network (CNN). Previous work has shown that deep learning [2][4] and CNN’s [1] can be successfully applied to natural language processing tasks. The final layer of the network is a softmax layer with three neurons, with the output of each corresponding to the probability that the given input review is positive, neutral, or negative.

TODO: What tools/library we used (Keras)

TODO: Why we picked this model architecture

<p align="center"><img src="./assets/architecture.png" alt="Model Architecture" height="800"/></p>

As to be expected, during training we initially came across significant overfitting. To address this we introduced:
- Dropout with a factor of 0.2 at the input layer. Additional dropout in the hidden layers was explored but this impacted the model's performance.
- L2 kernel and bias regularizers with a factor of 0.04 on the dense/fully-connected layers. We also considered adding regularizers to the convolutional layers, but we found them to be much more sensitive to regularization.
- Early stopping on training once the validation loss stops improving for 4 epochs. This allows us to avoid further overfitting and wasting time on excess iterations.

We spent a considerable amount of time tuning the architecture and hyperparameters of our model to achieve the desired accuracy levels. The best results were found with the Adam optimizer using a learning rate of 0.0002. Visualizations of loss vs epochs at this learning rate can be found in the Results section below.

### Results

To evaluate our model, we look at the accuracy with which movie reviews are assigned the correct labels. In our initial proposal, we hoped to achieve an accuracy of at least 70%, which approaches the results found in existing literature (mid 70% range with classical ML methods such as SVMs [3], above 80% with deep learning [2]). We ultimate were able to achieve a test accuracy of 70.658684% after tuning.

TODO: Compare to results from literature

The following are visualizations of how the loss and accuracies across the training and validation sets changed across epochs during training.

<p align="center"><img src="./assets/loss_graph.png" alt="Loss Graph"/></p>

<p align="center"><img src="./assets/accuracy_graph.png" alt="Accuracy Graph"/></p>

### Conclusion

### References

[1] Conneau, Alexis et al. “Very Deep Convolutional Networks for Natural Language Processing.” ArXiv abs/1606.01781 (2016): n. Pag.

[2] “Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.” Deeply Moving: Deep Learning for Sentiment Analysis, nlp.stanford.edu/sentiment/index.html.

[3] Parkhe, V., Biswas, B. Sentiment analysis of movie reviews: finding most important movie aspects using driving factors. Soft Comput 20, 3373–3379 (2016). https://doi.org/10.1007/s00500-015-1779-1

[4] Wu, Jean Y.. “Predicting Sentiment from Rotten Tomatoes Movie Reviews.” (2012).

[5] Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

[6] Pang, Bo et al. “Movie Review Data.” www.cs.cornell.edu/people/pabo/movie-review-data/.

[7] Pang, Bo et al. "Papers using our movie review data." http://www.cs.cornell.edu/people/pabo/movie-review-data/otherexperiments.html
