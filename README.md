# Topic detection and sentiment analysis of Google Play app reviews

This project has three parts -
1. Crawling Google Play to scrap app reviews
2. Creating a training set by humanly identifying bug related reviews in the training set and marking user sentiment
3. Train the algorithm using the training set and test it on a test set

The crawling part is still not complete. So far I can only crawl the first 40 reviews on an app page. The script needs to be modified to be Ajax crawlable.

The second part is straight forward but time consuming.

In the third part, I have used bag of words model and random forest. The script is largely taken from the Kaggle competition - https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words. I have modified the code to add two training criteria - bug related review and sentiment. I have also added Porter Stemming in the model.