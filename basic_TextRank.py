'''
basic_TextRank.py

Implements the basic TextRank algorithm for keyword selection
'''

#from __future__ import division
import re
import nltk
import numpy as np
from collections import defaultdict
import random

def keywords():
    with open("Example_texts/matrix_summary_long.txt", 'r') as f:
        text = f.read()

    M, word_indices = build_keyword_graph(text)

    v = PageRank(M)
    ordering = np.argsort(v)[::-1][:len(v)]

    print("Words in order of importance:\n")

    for i in range(len(ordering)):
        print(word_indices[ordering[i]], v[ordering[i]])

    keywords = [word_indices[ordering[i]] for i in range(len(ordering) // 8)]

    print("TextRank Keywords:")
    for k in keywords : print(k)


    D = rank_by_frequency(text)

    L = []

    for key in D:
        L.append((D[key], key))

    L.sort(reverse=True)

def sentence_rank():
    with open("Example_texts/inception_summary_wikipedia.txt", 'r') as f:
        text = f.read()

    M, sentences = build_sentence_graph(text)
    v = PageRank(M)
    ordering = np.argsort(v)[::-1][:len(v)]

    print("Sentences in order of importance:\n")
    for i in range(len(ordering)):
        print(sentences[ordering[i]] + '\n')

def rank_by_frequency(text):

    text_list = nltk.word_tokenize(text)
    tagged_list = nltk.pos_tag(text_list)

    D = defaultdict(int)

    for word, pos in tagged_list:
        if Accepted_part_of_speech(pos) and not word in {'(', ')'}:
            D[word] += 1

    return D

def build_keyword_graph(text):

    text_list = nltk.word_tokenize(text)
    tagged_text = nltk.pos_tag(text_list)

    index = 0
    possible_keywords = {}

    #tagged_text = filter(lambda (a,b): b in ACCEPTED_POS, tagged_text)

    for word, pos in tagged_text:
        if Accepted_part_of_speech(pos) and not word in {'(', ')'}: \
            #The nltk pos tagger sometimes includes parentheses
            if word not in possible_keywords:
                possible_keywords[word] = index
                index += 1

    nb_candidates = len(possible_keywords)
    window_size = 10

    #print possible_keywords

    M = np.zeros((nb_candidates, nb_candidates))
    for i in range(len(text_list)):

        word = text_list[i]

        if word not in possible_keywords:
            continue
        n = possible_keywords[word]
        M[n][n] = 6

        for j in range(-5,6):
            if j == 0:
                continue

            if i+j >= 0 and i+j < len(text_list):
                context_word = text_list[i+j]
                if context_word in possible_keywords:
                    m = possible_keywords[context_word]
                    M[m][n] = 6-abs(j)
                    M[n][m] = 6-abs(j)

    '''
    for a in possible_keywords:
        print possible_keywords[a], a
    '''

    ordered_words = ["" for i in range(nb_candidates)]

    for word in possible_keywords:
        ordered_words[possible_keywords[word]] = word

    M = M / np.sum(M, axis=0)

    return M, ordered_words

def build_sentence_graph(text):

    import nltk.data
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = sent_detector.tokenize(text)

    #sentence_index = {}
    nb_sentences = len(sentences)

    #for i in range(len(nb_sentences)):
    #  sentence_index[sentences[i]] = i

    M = np.zeros( (nb_sentences, nb_sentences) )

    for i in range(nb_sentences):
        for j in range(nb_sentences):
            M[i][j] = sentence_similarity( sentences[i], sentences[j] )
            M[j][i] = M[i][j]

    M = M / np.sum(M, axis=0)
    return M, sentences

def sentence_similarity(s1, s2):

    s1 = nltk.word_tokenize(s1)
    s2 = nltk.word_tokenize(s2)

    import math
    return len(set(s1) | set(s2)) / ( math.log(len(s1)) + math.log(len(s2)) )

def Accepted_part_of_speech(pos): #Accepting nouns and adjectives
    return pos in {'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS'}

def PageRank(M):

    n = M.shape[0]
    beta = 0.85

    e = np.ones((n,))

    v = e / n

    for i in range(50):
        v = beta*np.dot(M,v) + (1-beta)*e/n
        #print("vector after iteration %d" % i)
        #print(v)
        if abs(np.sum(v) - 1) > 0.01:
            print("SOMETHING IS WRONG")
            break

    return v

if __name__ == "__main__":
    keywords()
