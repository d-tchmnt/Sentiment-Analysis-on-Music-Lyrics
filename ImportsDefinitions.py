#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import re
from sklearn.externals import joblib 
import pickle
import numpy as np
from nltk.corpus import stopwords
import time
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree

from sklearn.svm import LinearSVC


import seaborn as sns

#accuracy_score(labels_test,pred)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
#accuracy_score(labels_test,pred)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from yellowbrick.text import FreqDistVisualizer
from yellowbrick.datasets import load_hobbies
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB


from yellowbrick.classifier import ClassificationReport
from yellowbrick.datasets import load_occupancy
from yellowbrick.text import FreqDistVisualizer
from yellowbrick.datasets import load_hobbies


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('vader_lexicon')


# In[1]:


#Ορισμός συναρτήσεων για να ελέξγχουμε τον καλύτερο vectorizer
def tfidf_test_simple(X_train,X_test,y_train,y_test,token_izer):
    if (token_izer=='1'):
        tfvect= TfidfVectorizer(tokenizer=tokenizer_preproccessor)
    elif (token_izer=='2'):
        tfvect= TfidfVectorizer(tokenizer=tokenizer_preproccessor_imdb)
    else:
        tfvect= TfidfVectorizer()
    tfidf_train = tfvect.fit_transform(X_train)
    tfidf_test = tfvect.transform(X_test)
    nb=MultinomialNB()
    #train the model and timing it
    #kanoume prediction gia to x_test_dtm
    
    # cross val score/ predict
    cvec_score = cross_val_score(nb, tfidf_train, y_train, cv=4 )
    feature_names = tfvect.get_feature_names()
    print("Number of features: {}".format(len(feature_names)))
    print("to accuracy tou TFIDF me NB einai {}".format(cvec_score.mean()))
    
    visualizer = FreqDistVisualizer(features=feature_names, orient='h')
    visualizer.fit(tfidf_train)
    visualizer.poof()
    
    return cvec_score.mean()

def countvect_test_simple(X_train,X_test,y_train,y_test,token_izer):
    if (token_izer=='1'):
        countvect= CountVectorizer(tokenizer=tokenizer_preproccessor)
    elif (token_izer=='2'):
        countvect= CountVectorizer(tokenizer=tokenizer_preproccessor_imdb)
    else:
        countvect= CountVectorizer()
    #CountVect
    countvect.fit(X_train)
    #to metatrepoume se dtm sparse matrix
    X_train_dtm=countvect.transform(X_train)
    X_test_dtm=countvect.transform(X_test)
    #Ftiaxnoume Multinomial Naive Bayes modelo
    nb=MultinomialNB()
    #kanoume prediction gia to x_test_dtm
    
    # cross val score/ predict
    cvec_score = cross_val_score(nb, X_train_dtm, y_train, cv=4 )
        
          
    feature_names = countvect.get_feature_names()
    print("Number of features: {}".format(len(feature_names)))
    print("to accuracy tou CountVectorizer me NB einai: {}".format(cvec_score.mean()))
    
    visualizer = FreqDistVisualizer(features=feature_names, orient='h')
    visualizer.fit(X_train_dtm)
    visualizer.poof()
    
    return cvec_score.mean()


# In[9]:


#oρισμός συναρτήσεων για να ελέγχουμε το max_df
def countvect_test_maxdf(X_train,X_test,y_train,y_test,token_izer,maxdf):
    if (token_izer=='1'):
        countvect= CountVectorizer(tokenizer=tokenizer_preproccessor,max_df=maxdf)
    elif (token_izer=='2'):
        countvect= CountVectorizer(tokenizer=tokenizer_preproccessor_imdb,max_df=maxdf)
    else:
        countvect= CountVectorizer(max_df=maxdf)
    X_train_dtm = countvect.fit_transform(X_train)
    X_test_dtm = countvect.transform(X_test)
    nb=MultinomialNB()
    cvec_score = cross_val_score(nb, X_train_dtm, y_train, cv=4 )
   
    #kanoume evaluate ta apotelesmata mas me Logistic Regression
    feature_names = countvect.get_feature_names()
    print("Number of features: {}".format(len(feature_names)))
    print("To accuracy NB me Max df: {} είναι : {} ".format(maxdf,cvec_score))
    return cvec_score.mean()


def tfidf_test_maxdf(X_train,X_test,y_train,y_test,token_izer,maxdf):
    if (token_izer=='1'):
        tfvect= TfidfVectorizer(tokenizer=tokenizer_preproccessor,max_df=maxdf)
    elif (token_izer=='2'):
        tfvect= TfidfVectorizer(tokenizer=tokenizer_preproccessor_imdb,max_df=maxdf)
    else:
        tfvect= TfidfVectorizer(max_df=maxdf)
    tfidf_train = tfvect.fit_transform(X_train)
    tfidf_test = tfvect.transform(X_test)
    nb=MultinomialNB()
    cvec_score = cross_val_score(nb, tfidf_train, y_train, cv=4 )
    
    #kanoume evaluate ta apotelesmata mas me Logistic Regression
    feature_names = tfvect.get_feature_names()
    print("Number of features: {}".format(len(feature_names)))
    print("To accuracy NB me max df: {} είναι : {} ".format(maxdf,cvec_score))
    return cvec_score.mean()


# In[12]:


def countvect_test_ngrams(X_train,X_test,y_train,y_test,token_izer,ngrams):
    if (token_izer=='1'):
        countvect= CountVectorizer(tokenizer=tokenizer_preproccessor,ngram_range=(1,ngrams))
    elif (token_izer=='2'):
        countvect= CountVectorizer(tokenizer=tokenizer_preproccessor_imdb,ngram_range=(1,ngrams))
    else:
        countvect= CountVectorizer(ngram_range=(1,ngrams))
    X_train_dtm = countvect.fit_transform(X_train)
    X_test_dtm = countvect.transform(X_test)
    nb=MultinomialNB()
    cvec_score = cross_val_score(nb, X_train_dtm, y_train, cv=4 )
   
    #kanoume evaluate ta apotelesmata mas me Logistic Regression
    feature_names = countvect.get_feature_names()
    print("Number of features: {}".format(len(feature_names)))
    print("To accuracy NB me {}-ngrams είναι : {} ".format(ngrams,cvec_score))
    return cvec_score.mean()

def tfidf_test_ngrams(X_train,X_test,y_train,y_test,token_izer,ngrams):
    if (token_izer=='1'):
        tfvect= TfidfVectorizer(tokenizer=tokenizer_preproccessor,ngram_range=(1,ngrams))
    elif (token_izer=='2'):
        tfvect= TfidfVectorizer(tokenizer=tokenizer_preproccessor_imdb,ngram_range=(1,ngrams))
    else:
        tfvect= TfidfVectorizer(ngram_range=(1,ngrams))
    tfidf_train = tfvect.fit_transform(X_train)
    tfidf_test = tfvect.transform(X_test)
    nb=MultinomialNB()
    cvec_score = cross_val_score(nb, tfidf_train, y_train, cv=4 )
    
    #kanoume evaluate ta apotelesmata mas me Logistic Regression
    feature_names = tfvect.get_feature_names()
    print("Number of features: {}".format(len(feature_names)))
    print("To accuracy NB me {}-ngrams είναι : {} ".format(ngrams,cvec_score))
    return cvec_score.mean()


# In[ ]:


def countvect_test_maxfeat(X_train,X_test,y_train,y_test,token_izer,maxfeat):  
    if (token_izer=='1'):
        countvect= CountVectorizer(tokenizer=tokenizer_preproccessor,max_features=maxfeat)
    elif (token_izer=='2'):
        countvect= CountVectorizer(tokenizer=tokenizer_preproccessor_imdb,max_features=maxfeat)
    else:
        countvect= CountVectorizer(max_features=maxfeat)
    X_train_dtm = countvect.fit_transform(X_train)
    X_test_dtm = countvect.transform(X_test)
    nb=MultinomialNB()
    cvec_score = cross_val_score(nb, X_train_dtm, y_train, cv=4 )
   
    #kanoume evaluate ta apotelesmata mas me Logistic Regression
    feature_names = countvect.get_feature_names()
    print("Number of features: {}".format(len(feature_names)))
    print("To accuracy NB me max features {} είναι : {} ".format(maxfeat,cvec_score))
    return cvec_score.mean()

def tfidf_test_maxfeat(X_train,X_test,y_train,y_test,token_izer,maxfeat):
    if (token_izer=='1'):
        tfvect= TfidfVectorizer(tokenizer=tokenizer_preproccessor,max_features=maxfeat)
    elif (token_izer=='2'):
        tfvect= TfidfVectorizer(tokenizer=tokenizer_preproccessor_imdb,max_features=maxfeat)
    else:
        tfvect= TfidfVectorizer(max_features=maxfeat)
    tfidf_train = tfvect.fit_transform(X_train)
    tfidf_test = tfvect.transform(X_test)
    nb=MultinomialNB()
    cvec_score = cross_val_score(nb, tfidf_train, y_train, cv=4 )
    
    #kanoume evaluate ta apotelesmata mas me Logistic Regression
    feature_names = tfvect.get_feature_names()
    print("Number of features: {}".format(len(feature_names)))
    print("To accuracy NB me max featurues : {} είναι : {} ".format(maxfeat,cvec_score))
    return cvec_score.mean()


# In[4]:


def tokenizer_preproccessor(text):
    stop = set(stopwords.words('english'))
    #kanoume preprocessing ta dedomena mas
    word_tokens=word_tokenize(text.lower()) #kanoume tokenize
    #print("Έχουμε " + str(len(word_tokens))+ " tokens")
    #filtered_word_tokens = [word for word in word_tokens if word not in stop] #svinoume ta stopwords
    #print("Αφού αφαιρέσαμε τα stopwords, έχουμε τελικά " + str(len(filtered_word_tokens)) + " tokens")
    #print(filtered_word_tokens)
    filtered_word_tokens = [re.sub(r'[^A-Za-z0-9]+', '', x) for x in word_tokens]  #svinoume ta punctuations
    #filtered_word_tokens = [x for x in filtered_word_tokens if x not in ['film','movie']]  #svinoume ta punctuations    
    filtered_word_tokens = [word for word in filtered_word_tokens if len(word)>1] #svinoume tis mikres lekseis
    #print("Αφού αφαιρέσαμε τα σημεία στίξης και τις μικρές λέξεις, έχουμε τελικά " + str(len(filtered_word_tokens)) + " tokens")
    #print(filtered_word_tokens)
    tagged_filtered_tokens=nltk.pos_tag(filtered_word_tokens) #kanoume pos tag tis lekseis gia syntaktiki analysi
    #print(tagged_filtered_tokens)
            #ftiaxnoume ta pos tags wste na mpoun san input sto sentisynset
    #epeidh ta dedomena tou postag ginontai tuples, ta metatrepoume se lista
    newtags=[] #lista me ta nea tags kai idio counter me ta tagged words
    for tag in tagged_filtered_tokens:
        if tag[1] in set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
            newtags.append('v')
        elif tag[1] in set(['JJ', 'JJR', 'JJS']):
             newtags.append('a')
        elif tag[1] in set(['RB', 'RBR', 'RBS']):
             newtags.append('r')
        elif tag[1] in set(['NNS', 'NN', 'NNP', 'NNPS']):
             newtags.append('n')
        else:
             newtags.append('a')
    
    lem_words=[] #edw tha mpoun oi lematized lekseis pou exoume kratisei apo to preprocessing
    counter=0 #vazoume ton counter gia na kanoume iterate ta stoixeia tis listas twn tags
    for word in tagged_filtered_tokens:    
        lem_words.append(wnl.lemmatize(word[0],newtags[counter]))
        counter+=1
    lem_words = [word for word in lem_words if word not in stop] #svinoume ta stopwords
    return lem_words

def tokenizer_preproccessor_imdb(text):
    stop = nltk.corpus.stopwords.words('english')
    stop.append('film')
    stop.append('movie')
    stop.append('br')
    #kanoume preprocessing ta dedomena mas
    word_tokens=word_tokenize(text.lower()) #kanoume tokenize
    #print(word_tokens)
    #print("Έχουμε " + str(len(word_tokens))+ " tokens")
    #filtered_word_tokens = [word for word in word_tokens if word not in stop] #svinoume ta stopwords
    #print("Αφού αφαιρέσαμε τα stopwords, έχουμε τελικά " + str(len(filtered_word_tokens)) + " tokens")
    #print(filtered_word_tokens)
    filtered_word_tokens = [re.sub(r'[^A-Za-z]+', '', x) for x in word_tokens]  #svinoume ta punctuations
    filtered_word_tokens = [x for x in filtered_word_tokens if ((x.startswith("'")==0 or x.startswith('-') or x.startswith('.')))]  #svinoume ta punctuations

    #filtered_word_tokens = [x for x in filtered_word_tokens if x not in ['film','movie']]  #svinoume ta punctuations    
    filtered_word_tokens = [word for word in filtered_word_tokens if len(word)>1] #svinoume tis mikres lekseis
    #print(filtered_word_tokens)
    #print("Αφού αφαιρέσαμε τα σημεία στίξης και τις μικρές λέξεις, έχουμε τελικά " + str(len(filtered_word_tokens)) + " tokens")
    #print(filtered_word_tokens)
   
    tagged_filtered_tokens=nltk.pos_tag(filtered_word_tokens) #kanoume pos tag tis lekseis gia syntaktiki analysi
    #print(tagged_filtered_tokens)
            #ftiaxnoume ta pos tags wste na mpoun san input sto sentisynset
    #epeidh ta dedomena tou postag ginontai tuples, ta metatrepoume se lista
    newtags=[] #lista me ta nea tags kai idio counter me ta tagged words
    for tag in tagged_filtered_tokens:
        if tag[1] in set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
            newtags.append('v')
        elif tag[1] in set(['JJ', 'JJR', 'JJS']):
             newtags.append('a')
        elif tag[1] in set(['RB', 'RBR', 'RBS']):
             newtags.append('r')
        elif tag[1] in set(['NNS', 'NN', 'NNP', 'NNPS']):
             newtags.append('n')
        else:
             newtags.append('a')
    
    lem_words=[] #edw tha mpoun oi lematized lekseis pou exoume kratisei apo to preprocessing
    counter=0 #vazoume ton counter gia na kanoume iterate ta stoixeia tis listas twn tags
    for word in tagged_filtered_tokens:    
        lem_words.append(wnl.lemmatize(word[0],newtags[counter]))
        counter+=1
    lem_words = [word for word in lem_words if word not in stop] #svinoume ta stopwords
    #print(lem_words)
    
    return lem_words


# In[5]:


def SentimentAnalysis_Sentiwordnet(text_samples):
    my_sentiments=[] #h lista pou tha periexei tosynaisthitiko score kathe keimenou
    my_sentiments_class=[]
    #print(stopwords)
    stop = set(stopwords.words('english'))
    for text in text_samples:
        #kanoume preprocessing ta dedomena mas
        word_tokens=word_tokenize(text.lower()) #kanoume tokenize
        #print("Έχουμε " + str(len(word_tokens))+ " tokens")
        filtered_word_tokens = [word for word in word_tokens if word not in stop] #svinoume ta stopwords
        #print("Αφού αφαιρέσαμε τα stopwords, έχουμε τελικά " + str(len(filtered_word_tokens)) + " tokens")
        #print(filtered_word_tokens)
        filtered_word_tokens = [re.sub(r'[^A-Za-z0-9]+', '', x) for x in filtered_word_tokens]  #svinoume ta punctuations
        filtered_word_tokens = [word for word in filtered_word_tokens if len(word)>1] #svinoume tis mikres lekseis
        #print("Αφού αφαιρέσαμε τα σημεία στίξης και τις μικρές λέξεις, έχουμε τελικά " + str(len(filtered_word_tokens)) + " tokens")
        #print(filtered_word_tokens)
        tagged_filtered_tokens=nltk.pos_tag(filtered_word_tokens) #kanoume pos tag tis lekseis gia syntaktiki analysi
        #print(tagged_filtered_tokens)
                #ftiaxnoume ta pos tags wste na mpoun san input sto sentisynset
        #epeidh ta dedomena tou postag ginontai tuples, ta metatrepoume se lista
        newtags=[] #lista me ta nea tags kai idio counter me ta tagged words
        for tag in tagged_filtered_tokens:
            if tag[1] in set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
                newtags.append('v')
            elif tag[1] in set(['JJ', 'JJR', 'JJS']):
                 newtags.append('a')
            elif tag[1] in set(['RB', 'RBR', 'RBS']):
                 newtags.append('r')
            elif tag[1] in set(['NNS', 'NN', 'NNP', 'NNPS']):
                 newtags.append('n')
            else:
                 newtags.append('a')
                    
                    

        lem_words=[] #edw tha mpoun oi lematized lekseis pou exoume kratisei apo to preprocessing
        counter=0 #vazoume ton counter gia na kanoume iterate ta stoixeia tis listas twn tags
        for word in tagged_filtered_tokens:    
            lem_words.append(wnl.lemmatize(word[0],newtags[counter]))
            counter+=1
           # print (newtags)
       # new_words_tags_dict = {'word':'synscore'}
        #print(newtags,lem_words)
            #print(lem_words)
        #ypologizoume to synaisthima kathe leksis , mazi me to POSTAG tis
        posscore=0
        negscore=0
        for i in range(len(lem_words)): 
            synsets = swn.senti_synsets(lem_words[i],newtags[i])
            for synst in synsets: #athroizoume ta thetika kai ta arnhtika score kathe leksis
                posscore=posscore+synst.pos_score()
                negscore=negscore+synst.neg_score()     
        my_sentiments.append(posscore-negscore)
        #print(my_sentiments)
        if (posscore-negscore)>=0:
            my_sentiments_class.append(1)
        else:
            my_sentiments_class.append(0)
    print(len(my_sentiments))
    return(my_sentiments_class)


# In[6]:


def SentimentAnalysis_Vader(text_samples):
    analyzer = SentimentIntensityAnalyzer()
    vader_sentiments=[]
    vader_class_sentiments=[]
    for text in text_samples:
        sum=0
        #Kovoume kathe keimeno se protaseis wste na vgalei sentiment polarity o vader, ta opoia athroizoume
        sentences=text.split('\n')   
        for sentence in sentences:
            sent = analyzer.polarity_scores(sentence)
            #print("{:-<65} {}".format(sentence, vs))
            sum=sum+sent['compound']
        average=sum/len(sentences)
        vader_sentiments.append(average)
        if (average>=0):
            vader_class_sentiments.append(1)       
        else:
            vader_class_sentiments.append(0)
        #print(sentiments)
        #print(average)
    #Data Examination for Unnotated Dataset
    print("Έχουμε " + str(vader_class_sentiments.count(1)) + " χαρούμενα τραγούδια")
    print("Έχουμε " + str(vader_class_sentiments.count(0)) + " στενάχωρα τραγούδια")
    return vader_class_sentiments


# In[ ]:


def classifier_finder(X_train,X_test,y_train,y_test):
    classifiers=[]

    scores=[]
    model1=LogisticRegression(max_iter=1000)
    classifiers.append(model1)
    model2=MultinomialNB()
    classifiers.append(model2)
    model4 = tree.DecisionTreeClassifier()
    classifiers.append(model4)
    model5 = RandomForestClassifier()
    classifiers.append(model5)
    model6=LinearSVC(max_iter=2000)
    classifiers.append(model6)
    
    
    for clf in classifiers:
        clf.fit(X_train, y_train)
        
        cvec_score = cross_val_score(clf, X_train, y_train, cv=10 )
        
        
        print("Η επιτυχία του  %s είναι:  %s"%(clf, cvec_score.mean()))
        scores.append(cvec_score.mean())
# DataFrame Accuracy 
    scores_df = pd.DataFrame()
    scores_df['params']= ['Logistisc Regression','Multinomial Naive Bayes','Decision Tree','Random Forest','Linear SVC']
    scores_df['scores']= scores
    print(scores_df)


# In[ ]:


def classifier_finder_music(X_train,y_train):
    classifiers=[]
    #COUNT VECTORIZER

    model1=LogisticRegression()
    classifiers.append(model1)
    model2 = tree.DecisionTreeClassifier()
    classifiers.append(model2)
    model3 = RandomForestClassifier()
    classifiers.append(model3)
    model4=LinearSVC()
    classifiers.append(model4)
    model5 = KNeighborsClassifier()
    classifiers.append(model5)
    scorelist=[]

    for clf in classifiers:
        clf.fit(X_train, y_train)
        cvec_score = cross_val_score(clf, X_train, y_train, cv=10)
        print("Τα αποτελέσματα του Cross-Validation για τον {} είναι:  {} ".format(clf,cvec_score.mean()))                                      
        scorelist.append(cvec_score.mean())
# DataFrame Accuracy 
    scores_df = pd.DataFrame()
    scores_df['params']= ['Logistisc Regression','Decision Tree','Random Forest','Linear SVC','K-nn']
    scores_df['scores']= scorelist
    print(scores_df)
    


# In[1]:


#δημιουργούμε συναρτήσεις για να αποθηκεύουμε και να φορτώνουμε  τις λίστες που θα φτιάξουμε με την ανάλυση του vader 
#και του sentiwordnet
def saveList(myList,filename):
    np.save(filename,myList)
    print("Saved successfully!")
    
def loadList(filename):
    tempNumpyArray=np.load(filename)
    return tempNumpyArray.tolist()


# In[ ]:




