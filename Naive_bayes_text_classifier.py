#!/usr/bin/env python
# coding: utf-8

# Made together with Emma Amundsen

# # Essential imports

# In[1]:


import pandas as pd
import numpy as np
import scipy as sp
#import csv
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

#import seaborn as sns  for visualisation of statistics

# for visualisation of graphs in web browser
get_ipython().run_line_magic('matplotlib', 'inline')


# # Natural Language ToolKit imports

# In[2]:


import nltk
from nltk.stem import LancasterStemmer

#from pandas_profiling import ProfileReport


# In[4]:


tweet_dataset = pd.read_csv('Tweets.csv')


# In[5]:


tweet_dataset['numerical_labels'] = tweet_dataset.airline_sentiment.map({'positive':1, 'neutral':0, 'negative':-1})


# In[6]:


tweet_dataset = tweet_dataset.drop(columns=['airline', 'tweet_location', 'user_timezone', 'tweet_created', 'tweet_coord',
                     'retweet_count', 'negativereason_gold', 'name', 'airline_sentiment_gold',
                     'negativereason', 'airline_sentiment_confidence', 
                     'negativereason_confidence', 'tweet_id'])


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(
    tweet_dataset['text'], tweet_dataset['numerical_labels'], random_state=5, test_size=0.2)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# In[10]:


# Code borrowed from https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d
# and modified to fit our code, author Cristhian Boujon
def get_top_n_words(corpus, vec, n=None):

    sum_words = corpus.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# # Class separator

# In[11]:


# Code borrowed from https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/ and modified to fit our code,
# author Jason Brownlee
# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    train = pd.DataFrame(X_train)
    separated = dict()
    
    # for loop going through each row in the dataset 
    for row in train.index:
        vector = dataset.loc[row]
        class_value = vector[-1]
        
        # makes a new key for a class in the dict
        if (class_value not in separated):
            separated[class_value] = list()
            
        # appends the row to the right class in the dict
        separated[class_value].append(row)
        
    return separated


# In[12]:


separate = separate_by_class(tweet_dataset)


# # Prior probability calculator

# In[13]:


def prior_prob(separate):
    dataset = X_train
    name = -2
    negative_prior = 0
    neutral_prior = 0
    positive_prior = 0
    
    for i in separate:
        total = len(separate[i]) / len(dataset)
        name = name + 1
        
        if name == -1:
            negative_prior = total
            print('negative:', negative_prior)
            
        elif name == 0:
            neutral_prior = total
            print('neutral:', neutral_prior)
            
        elif name == 1:
            positive_prior = total
            print('positive:', positive_prior) 
            
    return negative_prior, neutral_prior, positive_prior


# In[14]:


negative_prior, neutral_prior, positive_prior = prior_prob(separate)


# In[15]:


print(negative_prior, neutral_prior, positive_prior)


# # Gathering tweets by class

# In[16]:


def positive_total_tweets():
    train = pd.DataFrame(X_train)
    dataset = tweet_dataset
    positive_total = []
    
    for i in train.index:
        if dataset.loc[i].airline_sentiment == 'positive':
            positive_total.append(dataset.loc[i].text)
            
    return positive_total


# In[17]:


positive_total_tweets = positive_total_tweets()


# In[18]:


def negative_total_tweets():
    train = pd.DataFrame(X_train)
    dataset = tweet_dataset
    negative_total = []
    
    for i in train.index:
        if dataset.loc[i].airline_sentiment == 'negative':
            negative_total.append(dataset.loc[i].text)
            
    return negative_total


# In[19]:


negative_total_tweets = negative_total_tweets()


# In[20]:


def neutral_total_tweets():
    train = pd.DataFrame(X_train)
    dataset = tweet_dataset
    neutral_total = []
    
    for i in train.index:
        if dataset.loc[i].airline_sentiment == 'neutral':
            neutral_total.append(dataset.loc[i].text)
            
    return neutral_total


# In[21]:


neutral_total_tweets = neutral_total_tweets()


# # Word processing

# In[22]:


# Code found at https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1?fbclid=IwAR3T8kAfZmrLjiv7-iM5w1v-UDV5UxYbML9H-vyKOTav2NJ0Bbv7Jn6zCzs

def strip_emoji(text):
    RE_EMOJI = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    return RE_EMOJI.sub(r'', text)


# In[23]:


def strip_sentiment_list(data):
    lanc = LancasterStemmer()
    list_of_words = list()
    
    for i in data:
        new = re.sub(r'http\S+', '', i)
        
        for char in ['!', '?', ',', '.', '@', '#', '&', '\n', '"', '$', '%', "'", '(', ')', '*', '-', '+', '/', '^', 
                    '[', ']', '”', '_', ':', ';', '|', '{', '}', '~', '€', '£', '“', '1', '2', '3', '4', '5', '6', 
                    '7', '8', '9', '0', '=', '°', 'º', 'ʖ', '…', '⤵', '↔️']:
            new = new.replace(char, '') 
            
        new = strip_emoji(new)
        new = new.lower()
        new = new.split(" ")
        
        for j in new:
            new_word = lanc.stem(j)
            list_of_words.append(new_word)
            
    return strip_stopwords(list_of_words)


# In[24]:


def strip_sentiment(data):
    lanc = LancasterStemmer()
    list_of_words = list()
    
    new = re.sub(r'http\S+', '', data)
    
    for char in ['!', '?', ',', '.', '@', '#', '&', '\n', '"', '$', '%', "'", '(', ')', '*', '-', '+', '/', '^', 
                '[', ']', '”', '_', ':', ';', '|', '{', '}', '~', '€', '£', '“', '1', '2', '3', '4', '5', '6', 
                '7', '8', '9', '0', '=', '°', 'º', 'ʖ', '…', '⤵', '↔️']:
        new = new.replace(char, '') 
        
    new = strip_emoji(new)
    new = new.lower()
    new = new.split(" ")
    
    for j in new:
        new_word = lanc.stem(j)
        list_of_words.append(new_word)
        
    return strip_stopwords(list_of_words)


# In[25]:


def strip_stopwords(data):
    filtered = [word for word in data if not word in ENGLISH_STOP_WORDS]
    
    return filtered     


# In[26]:


vocab_strip = strip_sentiment_list(X_train)


# # Creates vocabulary

# In[27]:


def create_vocab(data):
    vocab = list()
    
    for word in data:
        
        if word not in vocab:
            vocab.append(word)
            
    return vocab


# In[28]:


vocab = create_vocab(vocab_strip)


# In[29]:


print(len(vocab))


# # Counting total nummber of words in each class

# In[30]:


positive_total_words = strip_sentiment_list(positive_total_tweets)
neutral_total_words = strip_sentiment_list(neutral_total_tweets)
negative_total_words = strip_sentiment_list(negative_total_tweets)


# In[31]:


print(len(positive_total_words), len(neutral_total_words), len(negative_total_words ))


# In[32]:


def total_words_dict(data):
    total_words_dict = dict()
    
    for row in data:
        
        if (row not in total_words_dict):
            total_words_dict[row] = 1
            
        total_words_dict[row] += 1
        
    return total_words_dict


# In[33]:


dict_positive = total_words_dict(positive_total_words)
dict_neutral = total_words_dict(neutral_total_words)
dict_negative = total_words_dict(negative_total_words)


# # Calculates the probability of a word to occurs in a class

# In[34]:


def conditional_probabilities(data, total_words, vocab_total=vocab):
    all_words = len(vocab_total)
    count = len(total_words)
    probs = dict()
    
    for key, value in data.items():
        
        if (key not in probs):
            probs[key] = (value + 1) / (count + all_words)
            
    return probs


# In[35]:


positive_words_prob = conditional_probabilities(dict_positive, positive_total_words)
neutral_words_prob = conditional_probabilities(dict_neutral, neutral_total_words)
negative_words_prob = conditional_probabilities(dict_negative, negative_total_words)


# # Calculates tweets probability for a single class

# In[36]:


def probability_for_class(dataset, class_word_prob, class_prob, class_total_words, vocab_total=vocab):
    
    all_total = len(vocab_total)
    
    if type(dataset) == list:
        
        list_of_prob_class = list()
        
        for row in dataset:
            prob = class_prob
            data = strip_sentiment(row)
            
            for word in data:
                try:
                    prob *= class_word_prob[word]
                    
                except:
                    prob *= (1 / (len(class_total_words) + all_total))
                    
            list_of_prob_class.append(prob)
            
        return list_of_prob_class
    
    elif type(dataset) == str:
        
        prob = class_prob
        data = strip_sentiment(dataset)
                               
        for word in data:
            try:
                prob *= class_word_prob[word]
                
            except:
                prob *= (1 / (len(class_total_words) + all_total))
                
        return prob
        


# # Compares predictions of class

# In[37]:


def highest_prob_class(positive_prob, neutral_prob, negative_prob):
    prediction = None
    if type(positive_prob) == list:
        prediction = list()
        
        for i in range(len(positive_prob) - 1):
            highest_prob = max([positive_prob[i], neutral_prob[i], negative_prob[i]])
            
            if highest_prob == positive_prob[i]:
                prediction.append(1)
                
            elif highest_prob == neutral_prob[i]:
                prediction.append(0)
            
            else:
                prediction.append(-1)
    
    else:
        
            highest_prob = max([positive_prob, neutral_prob, negative_prob])
            
            if highest_prob == positive_prob:
                prediction = 1
                
            elif highest_prob == neutral_prob:
                prediction = 0
            
            else:
                prediction = -1
        
    return prediction


# In[38]:


prediction_train = highest_prob_class(probability_for_class(list(X_train), positive_words_prob, positive_prior, positive_total_words), probability_for_class(list(X_train), neutral_words_prob, neutral_prior, neutral_total_words), probability_for_class(list(X_train), negative_words_prob, negative_prior, negative_total_words))


# In[39]:


prediction = highest_prob_class(probability_for_class(list(X_test), positive_words_prob, positive_prior, positive_total_words), probability_for_class(list(X_test), neutral_words_prob, neutral_prior, neutral_total_words), probability_for_class(list(X_test), negative_words_prob, negative_prior, negative_total_words))


# # Compares prediction with output

# In[40]:


def compare(test_input, test_output):
    error = list()
    
    for i in range(len(test_input) - 1):
        
        if test_input[i] == test_output[i]:
            error.append('correct')
            
        else:
            error.append('wrong')
            
    return error


# In[41]:


error_rating = compare(prediction, list(y_test))


# In[42]:


error_train = compare(prediction_train, list(y_train))


# # Score based on nummber of right outputs

# In[43]:


def score(error_rating):
    correct = 0
    wrong = 0
    
    for i in error_rating:
        
        if i == 'correct':
            correct += 1
            
        else:
            wrong += 1
            
    return correct / len(error_rating)


# In[44]:


score_test = score(error_rating)


# In[45]:


score_train = score(error_train)


# In[46]:


def single_tweet(data):
    prediction = highest_prob_class(probability_for_class(data, positive_words_prob, positive_prior, positive_total_words), probability_for_class(data, neutral_words_prob, neutral_prior, neutral_total_words), probability_for_class(data, negative_words_prob, negative_prior, negative_total_words))
    
    if prediction == -1:
        return 'Negative'
    
    elif prediction == 0:
        return 'Neutral'
    
    elif prediction == 1:
        return 'Positive'


# In[79]:


def explination_prob(data, class_word_prob):
    string = strip_sentiment(data)
    for word in string:
        if word in class_word_prob.keys():
            print('{:<10}'.format(word), 'has probablilty', '{:.5f}'.format(class_word_prob[word]), 'for this class')
        else:
            print(word, 'does not appare in the training data')
    


# In[77]:


def explination(data, positive = positive_words_prob, neutral = neutral_words_prob, negative = negative_words_prob):
    
    prediction = single_tweet(data)
    
    print(prediction, 'is the class which has been predicted for this tweet \n')
    
    if prediction == 'Positive':
        explination_prob(data, positive)
    elif prediction == 'Neutral':
        explination_prob(data, neutral)
    else:
        explination_prob(data, negative)


# In[80]:


explination("""@united once he found out we had a problem he avoided me like the plague. Was told "we can't find a supervisor." """)

