### Introduction

Sentiment analysis is the application of natural language processing (NLP) to learn and determine the emotion or feeling associated with a statement. This feeling can be categorized as positive, neutral or negative; sad, angry, or happy; as a rating from 1 to 10; etc.

Our team performs supervised learning with a neural network model to perform sentiment analysis on movie reviews. The corpus consists of reviews from multiple movie review aggregators such as IMDB and Rotten Tomatoes with corresponding sentiment labels, as found in existing datasets [5][6].

We first train a sentiment analysis model on the reviews to predict positive, neutral or negative sentiments. For further analysis, we average the sentiment results of individual movies to predict numerical movie ratings. We finally compare our sentiment analysis model’s predicted ratings to the actual ratings to determine whether there are biases associated with different movie review aggregators.

### Methods

The input data consists of labelled plaintext movie reviews from multiple movie review aggregators such as IMDB and Rotten Tomatoes. We use a word embedding tool such as word2vec to map the plaintext words contained in each review to vectors, which are merged to form a feature set representation. Our model categorizes reviews as positive, neutral, or negative, outputted as 3-element vectors ([1, 0, 0], [0, 1, 0], and [0, 0,1], respectively) from a linear layer. Based on the loss between these outputs and the correct sentiment labels, the model is optimized to predict correctly.

The sentiment analysis model may use a convolutional neural network (CNN). Previous work has shown that deep learning [2][4] and CNN’s [1] can be successfully applied to natural language processing tasks. The final layer of the network is a softmax layer with three neurons, with the output of each corresponding to the probability that the given input review is positive, neutral, or negative.


### Potential Results

We will evaluate the accuracy of our model by comparing the model’s predicted rating for each movie review aggregator to the actual rating. We hope to achieve an accuracy of at least 70%, which approaches the results found in existing literature (mid 70% range with classical ML methods such as SVMs [3], above 80% with deep learning [2]).

### Discussion

There are inherent problems with our approach. For example, individual reviews may be rated from zero to five stars, but we will classify each review only as positive, neutral, or negative. Hypothetically, if every review corresponded to four stars, our classification would cause our model to consider every review positive and predict a rating of five stars. Realistically, however, this problem should not be that significant since the actual reviews should be distributed over a wider range.

---

[GitHub page editor](https://github.com/baranusluel/sentiment-cnn/edit/master/index.md)
[Markdown cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
[GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/)

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/baranusluel/sentiment-cnn/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
