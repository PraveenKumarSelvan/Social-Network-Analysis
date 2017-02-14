# coding: utf-8
#Recommendation systems

from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ 
    Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ 
    Tokenize String
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    tokens=[]
    for row in movies['genres']:
        tokens.append(tokenize_string(row))
    movies['tokens'] = tokens
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    features=[]
    cnt = Counter()
    N=len(movies.index)
    df = defaultdict(int)
    tf = defaultdict()
    vocab_list=[]
    vocab={}
    for  index,row in movies.iterrows():
        terms = defaultdict(int)
        genre=row['genres']
        movie_id=row['movieId']
        i=genre.split('|')
        #print(i)
        temp_list=[]
        for items in i:
            if items not in vocab_list:
                vocab_list.append(items)
            if items not in temp_list:
                temp_list.append(items)
                df[items]+=1
            terms[items] += 1
        tf[movie_id]=terms
    list=sorted(vocab_list)
    i=0
    for list in list:
        vocab[list]=i
        i+=1
    for index, row in movies.iterrows():
        indptr = [0]
        indices = []
        data = []
        tfidf = []
        val_dict=tf[row['movieId']]
        genre = row['genres']
        movie_id = row['movieId']
        i = genre.split('|')
        max_name, max_val = max(val_dict.items(), key=lambda x: x[1])
        temp_list=[]
        for items in i:
            df_term=df[items]
            term=val_dict[items]
            total= (term/max_val) * math.log((N/df_term), 10)
            #print(items)
            #print(total)
            if items in vocab:
                if items not in temp_list:
                    temp_list.append(items)
                    index = vocab[items]
                    indices.append(index)
                    data.append(total)
        indptr.append(len(indices))
        x = csr_matrix((data, indices, indptr),shape=(1,len(vocab)))
        features.append(x)
        #print(x.toarray())
    movies['features']=features
    return movies,vocab

def train_test_split(ratings):
    """
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    val= float(math.sqrt(a.multiply(a).asfptype().sum(axis=None)) * math.sqrt(b.multiply(b).asfptype().sum(axis=None)))
    return float((a.multiply(b).sum(axis=None))/val)
    #dot(a,b.transpose).
    #return (a*(b.transpose())/ (np.linalg.norm(a)* np.linalg.norm(b)))
    #/ ((np.linalg.norm(a))*(np.linalg.norm(b)))


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    all_predictions=[]
    for index,row in ratings_test.iterrows():
        user=row['userId']
        movie_toberated=row['movieId']
        cos1=movies['features'].loc[movies['movieId'] == movie_toberated].iloc[0]
        all_rating=0
        predict_temp=0
        weight=0
        count=0
        df=ratings_train[ratings_train['userId'] == user]
        for index2,row2 in df.iterrows():
           
            movie_tobecompared=row2['movieId']
            cos2=movies['features'].loc[movies['movieId'] == movie_tobecompared].iloc[0]
            rating = row2['rating']
            cos=cosine_sim(cos1,cos2)
            all_rating=all_rating+rating
            count=count+1
            if(cos>0):
                predict_temp=predict_temp+(cos*rating)
                weight=weight+cos
        if(predict_temp>0):
            pred_rating=predict_temp/weight
        else:
            pred_rating=all_rating/count
        all_predictions.append(pred_rating)

    return (np.array(all_predictions))



def mean_absolute_error(predictions, ratings_test):
    """
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])




if __name__ == '__main__':
    main()
