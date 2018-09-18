# coding: utf-8

# In[8]:

import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.grid_search import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.naive_bayes import MultinomialNB


# In[45]:

# read the labels and bigram words frequency
labels=pd.read_excel("categorizing info.xlsx",header=None)
labels.columns=["filename","gender","age","others"]
age=labels["age"].as_matrix()
gender=labels["gender"].as_matrix()
print(age[:5])
print(gender[:5])
balance_ratio=math.floor(0.07*len(labels))
balance_ratio


# In[46]:


dt=DecisionTreeClassifier()
param_grid={'max_depth':range(1,51)}




def plotAccuracy_dt(rs,xlabel,title):
    # get the information from the grid research
    depths=list()
    scores = list()
    scores_std = list()
    for e in rs.grid_scores_:
        depths.append(e.parameters.get("max_depth"))
        scores.append(e.mean_validation_score)
        scores_std.append(np.std(e.cv_validation_scores))

    # plot out the mean accuracy and standard errors from cross validation of different alphas

    plt.figure().set_size_inches(13, 11)
    plt.semilogx(depths, scores)

    # plot error lines showing +/- std. errors of the scores
    std_error = scores_std / np.sqrt(3)

    plt.semilogx(depths, scores + std_error, 'b--')
    plt.semilogx(depths, scores - std_error, 'b--')

    # alpha=0.2 controls the translucency of the fill color
    plt.fill_between(depths, scores + std_error, scores - std_error, alpha=0.2)
    plt.ylabel('CV score +/- std error',fontsize=15)
    plt.xlabel(xlabel,fontsize=15)
    plt.title(title,fontsize=20)
    plt.axhline(np.max(scores), linestyle='--', color='.5')
    plt.xlim([depths[0], depths[-1]])
    #plt.ylim([ np.min(scores), np.max(scores)])
    #plt.vlines(rs.best_estimator_.max_depth,ymin=linestyle='dashed')
    fn=title+".png"
    plt.savefig(fn)

# get to the char data
bigram_char_tdidf=pd.read_csv("arraychartfidf.csv",header=None,lineterminator=";")
#split the train test data
char_train, char_test, age_train, age_test = train_test_split(bigram_char_tdidf, age, test_size=0.2, random_state=42)
# testing different depth on age
rsearch.fit(char_train,age_train)
# summarize the results of the random parameter search
print("dt char whole best score on age: ",rsearch.best_score_)
print("dt char whole best depth on age: ",rsearch.best_estimator_.max_depth)
dt=DecisionTreeClassifier(max_depth=rsearch.best_estimator_.max_depth)
dt.fit(char_test,age_test)
print("dt char whole test score on age: ",dt.score(char_test,age_test))
plotAccuracy_dt(rsearch,"maximum depth of the tree","Decision Tree + char bigram + whole dataset")


# In[ ]:




# In[ ]:

# do the char balanced tdidf dataset on decision tree
bigram_char_tdidf["age"]=age
bigram_char_tdidf_10s=bigram_char_tdidf[bigram_char_tdidf["age"]=="10s"][:balance_ratio]
bigram_char_tdidf_20s=bigram_char_tdidf[bigram_char_tdidf["age"]=="20s"][:balance_ratio]
bigram_char_tdidf_30s=bigram_char_tdidf[bigram_char_tdidf["age"]=="30s"][:balance_ratio]
# bigram_word_tdidf_balanced will be the new dataset after balanced
bigram_char_tdidf_balanced=pd.concat([bigram_char_tdidf_10s,bigram_char_tdidf_20s,bigram_char_tdidf_30s],axis=0)
age_balance=bigram_char_tdidf["age"].as_matrix()
del bigram_char_tdidf_balanced["age"]


# In[ ]:

#split the train test data
char_train, char_test, age_train, age_test = train_test_split(bigram_char_tdidf_balanced,age_balance, test_size=0.2, random_state=42)
# testing different depth on age
rsearch.fit(char_train,age_train)
# summarize the results of the random parameter search
print("dt char balanced best score on age: ",rsearch.best_score_)
print("dt char balanced best depth on age: ",rsearch.best_estimator_.max_depth)
dt=DecisionTreeClassifier(max_depth=rsearch.best_estimator_.max_depth)
dt.fit(char_test,age_test)
print("dt char balanced test score on age: ",dt.score(char_test,age_test))
plotAccuracy_dt(rsearch,"maximum depth of the tree","Decision Tree + char bigram + balanced dataset")


# In[ ]:

del bigram_char_tdidf
del bigram_char_tdidf_balanced
del bigram_char_tdidf_balanced_10s
del bigram_char_tdidf_balanced_20s
del bigram_char_tdidf_balanced_30s

# In[ ]:




# In[82]:

# begin naive bayes
# do the word
bigram_word=pd.read_csv("arrayword.csv",header=None)
len(bigram_word)
#len(bigram_char_tdidf)
len(age)
len(gender)


# In[ ]:


def plotAccuracy_nb(rs,xlabel,title):
    # get the information from the grid research
    depths=list()
    scores = list()
    scores_std = list()
    for e in rs.grid_scores_:
        depths.append(e.parameters.get("alpha"))
        scores.append(e.mean_validation_score)
        scores_std.append(np.std(e.cv_validation_scores))

    # plot out the mean accuracy and standard errors from cross validation of different alphas

    plt.figure().set_size_inches(13, 11)
    plt.semilogx(depths, scores)

    # plot error lines showing +/- std. errors of the scores
    std_error = scores_std / np.sqrt(3)

    plt.semilogx(depths, scores + std_error, 'b--')
    plt.semilogx(depths, scores - std_error, 'b--')

    # alpha=0.2 controls the translucency of the fill color
    plt.fill_between(depths, scores + std_error, scores - std_error, alpha=0.2)
    plt.ylabel('CV score +/- std error',fontsize=15)
    plt.xlabel(xlabel,fontsize=15)
    plt.title(title,fontsize=20)
    plt.axhline(np.max(scores), linestyle='--', color='.5')
    plt.xlim([depths[0], depths[-1]])
    #plt.ylim([ np.min(scores), np.max(scores)])
    #plt.vlines(rs.best_estimator_.max_depth,ymin=linestyle='dashed')
    fn=title+".png"
    plt.savefig(fn)


# In[ ]:

#from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
param_grid={"alpha":np.array([0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])}
rsearch = RandomizedSearchCV(estimator=nb, param_distributions=param_grid,n_iter=len(param_grid))


# In[72]:


#split the train test data
word_train, word_test, age_train, age_test = train_test_split(bigram_word, age, test_size=0.2, random_state=42)
# testing different depth on age

rsearch.fit(word_train,age_train)
# summarize the results of the random parameter search
print("nb word whole best score on age: ",rsearch.best_score_)
print("nb word whole best depth on age: ",rsearch.best_estimator_.alpha)
nb=MultinomialNB(alpha=rsearch.best_estimator_.alpha)
nb.fit(word_test,age_test)
print("nb word whole test score on age: ",nb.score(word_test,age_test))
plotAccuracy_nb(rsearch,"Laplace smoothing parameter","NaiveBayes + word bigram + whole dataset")


# In[ ]:




# In[ ]:

# balance the dataset
bigram_word["age"]=age
bigram_word_10s=bigram_word[bigram_word["age"]=="10s"][:balance_ratio]
bigram_word_20s=bigram_word[bigram_word["age"]=="20s"][:balance_ratio]
bigram_word_30s=bigram_word[bigram_word["age"]=="30s"][:balance_ratio]
# bigram_word_tdidf_balanced will be the new dataset after balanced
bigram_word_balanced=pd.concat([bigram_word_10s,bigram_word_20s,bigram_word_30s],axis=0)
age_balance=bigram_word_balanced["age"].as_matrix()
del bigram_word_balanced["age"]

#split the train test data
word_train, word_test, age_train, age_test = train_test_split(bigram_word_balanced,age_balance, test_size=0.2, random_state=42)
# testing different depth on age
rsearch.fit(word_train,age_train)
# summarize the results of the random parameter search
print("nb word balanced best score on age: ",rsearch.best_score_)
print("nb word balanced best depth on age: ",rsearch.best_estimator_.alpha)
nb=MultinomialNB(alpha=rsearch.best_estimator_.alpha)
nb.fit(word_test,age_test)
print("nb word balanced test score on age: ",nb.score(word_test,age_test))
plotAccuracy_nb(rsearch,"Laplace smoothing parameter","NaiveBayes + word bigram + balanced dataset")

del bigram_word


# In[ ]:




# In[ ]:

# do the char
bigram_char=pd.read_csv("arraychar.csv",header=None,lineterminator=";")

#split the train test data
char_train, char_test, age_train, age_test = train_test_split(bigram_char, age, test_size=0.2, random_state=42)
# testing different depth on age

rsearch.fit(char_train,age_train)
# summarize the results of the random parameter search
print("nb char whole best score on age: ",rsearch.best_score_)
print("nb char whole best depth on age: ",rsearch.best_estimator_.alpha)
nb=MultinomialNB(alpha=rsearch.best_estimator_.alpha)
nb.fit(char_test,age_test)
print("nb word whole test score on age: ",nb.score(char_test,age_test))
plotAccuracy_nb(rsearch,"Laplace smoothing parameter","NaiveBayes + char bigram + whole dataset")


# In[ ]:

# balance the dataset
bigram_char["age"]=age
bigram_char_10s=bigram_char[bigram_char["age"]=="10s"][:balance_ratio]
bigram_char_20s=bigram_char[bigram_char["age"]=="20s"][:balance_ratio]
bigram_char_30s=bigram_char[bigram_char["age"]=="30s"][:balance_ratio]
# bigram_char_balanced will be the new dataset after balanced
bigram_char_balanced=pd.concat([bigram_char_10s,bigram_char_20s,bigram_char_30s],axis=0)
age_balance=bigram_char_balanced["age"].as_matrix()
del bigram_char_balanced["age"]

#split the train test data
char_train, char_test, age_train, age_test = train_test_split(bigram_char_balanced, age_balance, test_size=0.2, random_state=42)
# testing different depth on age

rsearch.fit(char_train,age_train)
# summarize the results of the random parameter search
print("nb char _balanced best score on age: ",rsearch.best_score_)
print("nb char _balanced best depth on age: ",rsearch.best_estimator_.alpha)
nb=MultinomialNB(alpha=rsearch.best_estimator_.alpha)
nb.fit(char_test,age_test)
print("nb word _balanced test score on age: ",nb.score(char_test,age_test))
plotAccuracy_nb(rsearch,"Laplace smoothing parameter","NaiveBayes + char bigram + balanced dataset")

