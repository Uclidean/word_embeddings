import matplotlib.pyplot as plt
import numpy as np
import random
import scipy as sp
from sklearn.decomposition import TruncatedSVD

def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1
    corpus_words = sorted(list(set(word for doc in corpus for word in doc)))
    num_corpus_words = len(corpus_words)
    return corpus_words, num_corpus_words

def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).

        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
            number of co-occurring words.
            
            For example, if we take the document "START All that glitters is not gold END" with window size of 4,
            "All" will co-occur with "START", "that", "glitters", "is", and "not".

        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = None
    word2Ind = {}

    M = np.zeros((num_words, num_words)) # Co-occurence matrix
    word2Ind = {word: ind for ind, word in enumerate(words)} # Dict that maps a given word to some index for M
    for doc in corpus: # Iterate through all documents in corpus
        for i in range(len(doc)): # Iterate through all words in the doc
            row_ind = word2Ind[doc[i]] # Get index of word we're looking at
            window = list(range(max(0, i-window_size), min(len(doc), i+window_size+1))) # Context words
            window.remove(i) # Window should only consist of context words
            col_inds = [word2Ind[doc[j]] for j in window] # Get indeces of all context words
            for col_ind in col_inds:
                M[row_ind, col_ind] +=1 
    return M, word2Ind

def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        Params:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """    
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    svd = TruncatedSVD(n_components=k, n_iter=n_iters)
    M_reduced = svd.fit_transform(M)
    print("Done.")
    return M_reduced

def plot_embeddings(M_reduced, word2Ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2Ind.
        Include a label next to each point.
        
        Params:
            M_reduced (numpy matrix of shape (number of unique words in the corpus , k)): matrix of k-dimensioal word embeddings
            word2Ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """

    x_coords = M_reduced[:, 0]
    y_coords = M_reduced[:, 1]

    for word in words:
        ind = word2Ind[word]
        (x, y) = M_reduced[ind]
        plt.scatter(x, y, marker='x', color='blue')
        plt.text(x, y, word, fontsize=9)
