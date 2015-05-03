## Majority of code is taken from
## https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
## I have added porter stemming and modified the code to detect
## bug related app reviews and predict positive or negative sentiment 

import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *


train = pd.read_csv("review_train_2.csv",header=0)
print train.shape

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()

    
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    b=[]
    stemmer = PorterStemmer()
    for word in meaningful_words:
        b.append(stemmer.stem(word))
	
    return( " ".join( b ))  

##print clean_review
##clean_review = review_to_words( train["review_text"][0] )


# Get the number of reviews based on the dataframe column size
num_reviews = train["review_text"].size
print num_reviews

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

print "Cleaning and parsing the training set movie reviews...\n"

for i in xrange( 0, num_reviews ):
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % ( i+1, num_reviews ) 
    clean_train_reviews.append( review_to_words( train["review_text"][i] ))


print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

print train_data_features.shape

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print vocab

import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag
    
print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest classifiers with 200 trees
forest_bug = RandomForestClassifier(n_estimators = 200)
forest_sentiment = RandomForestClassifier(n_estimators = 200) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest_bug = forest_bug.fit( train_data_features, train["error_related"] )
forest_sentiment = forest_sentiment.fit( train_data_features, train["sentiment"] )

print forest_bug
print forest_sentiment


# Read the test data
test = pd.read_csv("review.csv", header=0)

# Verify that there are 25,000 rows and 2 columns
print test.shape

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review_text"])
clean_test_reviews = [] 

print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    clean_review = review_to_words( test["review_text"][i] )
    clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
bug_related = forest_bug.predict(test_data_features)
sentiment = forest_sentiment.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column,
# a "bug_related" column, a "sentiment" column and an "app" column
output = pd.DataFrame( data={"review_id":test["review_id"], "bug_related":bug_related, "sentiment":sentiment, "app": test["app_link"]} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )
