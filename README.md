# Sentiment Analysis with Convolutional Neural Networks
#### Baran Usluel, Sylesh K Suresh, Kenneth William Wardlaw, Vietfu Tang, and Yash Raghavendra Vaidya

We develop a deep convolutional neural network to perform sentiment analysis on movie reviews and
classify them as negative/neutral/positive. We achieved a test accuracy of 70.658684%.

View our final report here: https://baranusluel.com/sentiment-cnn/

Our code is contained in the Juypter notebook `project.ipynb`. We have included
a saved model `Sentiment_CNN_V1.h5` which yielded our best test results.

The notebook was executed on Google Colab, but it can be run locally as well. The path variables will
need to be adjusted accordingly.

### Dependencies

This project uses Keras with a Tensorflow backend. It also depends on numpy and scikit-learn.

The pretrained word2vec model (`GoogleNews-vectors-negative300.bin.gz`) will need to be downloaded.
It can be found [here](https://github.com/mmihaltz/word2vec-GoogleNews-vectors).

The dataset of movie reviews called `scale dataset v1.0` should also be downloaded
from [here](http://www.cs.cornell.edu/people/pabo/movie-review-data/) and unzipped.
