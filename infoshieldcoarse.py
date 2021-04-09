#!/usr/bin/env python3
# Author:   Catalina Vajiac
# Purpose:  Coarse clustering of text documents
# Usage:    ./infoshieldcoarse.py [filename]

import heapq
import math
import networkx as nx
import numpy as np
import os, sys
import pandas
import pickle
import random
import re
import scipy
import time

import matplotlib.pyplot as plt

from collections import Counter, defaultdict
from datetime import datetime
from networkx.algorithms import bipartite
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import adjusted_rand_score, homogeneity_score


# Utilities

def term_frequency(phrase, document):
    ''' return tf of phrase in document '''
    return sum([p == phrase for p in document])


def filter_text(text):
    if type(text) is not str: # nan
        return ''

    replace = [(r'\d+', ''), (r'[^\x00-\x7F]+', ' '), (re.compile('<.*?>'), '')]
    nbsp_variants = [('&nbsp;', ''), ('nbsp;', '')]
    br_variants = [('<b', ''), ('<br', ''), ('br>', '')]
    for source, target in replace + nbsp_variants + br_variants:
        text = re.sub(source, target, text)

    return text


class InfoShieldCoarse():
    def __init__(self, filename: str, doc_text_header='', doc_id_header='', num_phrases=10):
        # init basic variables
        self.time = time.time()
        self.num_phrases = num_phrases
        self.filename_full = filename.split('.')[0]
        self.filename = os.path.basename(filename).split('.')[0]
        #self.time_filename = '{}_streaming-{}_time.txt'.format(self.filename)
        self.ngrams = (5, 5)
        self.index_to_docid = Counter()
        self.docid_to_index = Counter()

        self.data = pandas.read_csv(filename, lineterminator='\n')
        self.determine_header_names(doc_text_header, doc_id_header)
        #self.data = self.data.drop_duplicates(subset=['title', self.description])

        if 'timestamp' in self.data.columns:
            self.data.sort_values(by=['timestamp']) # since ads not in order as they should be
        self.num_ads = len(self.data.index)
        self.cluster_graph = nx.Graph()

        # setup tfidf
        tfidf = TfidfVectorizer(token_pattern=r'[^\s]+', lowercase=False, ngram_range=self.ngrams,
                sublinear_tf=True)
        self.tokenizer = tfidf.build_analyzer()
        self.document_freq = defaultdict(float)
        self.term_freq = defaultdict(lambda: Counter())
        self.length = Counter()
        self.data[self.description] = self.data[self.description].apply(filter_text)
        self.tfidfs = tfidf.fit_transform(self.data[self.description])
        self.tfidf_indices = tfidf.get_feature_names()
        self.num_ads_features = 0


    def determine_header_names(self, doc_text_header: str, doc_id_header: str):
        ''' automatically determine relevant header names for doc id, doc text'''
        columns = set(self.data.columns)
        indices = {'ad_id', 'index', 'TweetID', 'id'}
        descriptions = {'u_Description', 'description', 'body', 'Tweet', 'text'}
        phones = {'u_PhoneNumbers', 'phone', 'PhoneNumber'}
        descriptions.add(doc_text_header)
        indices.add(doc_id_header)
        indices.add(doc_text_header)
        for name, field in [('text', descriptions), ('unique id', indices)]:#, ('phone #', phones)]:
            if not len(columns.intersection(field)):
                print('Add "{}" header to possible descriptions!'.format(name))
                exit(1)
        self.description = columns.intersection(descriptions).pop()
        self.id = columns.intersection(indices).pop()
        self.phone = columns.intersection(phones).pop() if len(columns.intersection(phones)) else ''


    def tokenize_text(self, text):
        #include_field = lambda x: x in self.data.columns and type(row[x]) == str
        return self.tokenizer(filter_text(text))


    def top_tfidf_phrases(self, doc_id, index: int, return_all=False):
        ''' return the top phrases with highest tfidf score '''
        #def score(doc_id: str, phrase: str) -> float:
        #    return self.term_freq[doc_id][phrase] * self.length[doc_id] / self.document_freq[phrase]
        #tfidf_pairs = [(score(doc_id, phrase), phrase) for phrase in phrases]
        _, cols = self.tfidfs[index].nonzero()
        tfidf_pairs = [(self.tfidfs[index, c], self.tfidf_indices[c]) for c in cols]
        return heapq.nlargest(self.num_phrases, tfidf_pairs)


    def process_ad(self, index: int, row):
        ''' find top phrases and add the ad to the cluster graph '''
        doc_id = row[self.id]
        self.index_to_docid[index] = doc_id
        self.docid_to_index[doc_id] = index

        if 'title' in row:
            text = row['title'] if type(row['title']) == str else ''
        else:
            text = ''

        text += ' ' + row[self.description] if type(row[self.description]) == str else ''
        phrases = self.tokenize_text(text)

        #top_tfidf = [phrase for _, phrase in self.top_tfidf_phrases(doc_id, set(phrases))]
        top_tfidf = [phrase for _, phrase in self.top_tfidf_phrases(doc_id, index)]


        self.cluster_graph.add_nodes_from(top_tfidf, bipartite=0)
        self.cluster_graph.add_node(doc_id, bipartite=1)
        self.cluster_graph.add_edges_from([(doc_id, phrase) for phrase in top_tfidf])


    def generate_labels(self):
        document_nodes = set([n for n, d in self.cluster_graph.nodes(data=True) if d['bipartite']])
        self.labels = [-1]*len(self.data.index)
        for i, component in enumerate(nx.connected_components(self.cluster_graph)):
            docs = [c for c in component if c in document_nodes]
            if len(docs) == 1:
                continue

            for docid in docs:
                self.labels[self.docid_to_index[docid]] = i


    def write_cluster_graph(self):
        ''' write cluster graph as pkl file '''
        if not os.path.isdir('pkl_files'):
            os.mkdir('pkl_files')

        #with open('pkl_files/{}_ad_graph.pkl'.format(self.filename), 'wb') as f:
        #    pickle.dump(self.cluster_graph, f)


    def write_csv_labels(self):
        ''' write new csv, with LSH labels '''
        filename_stub = self.filename_full
        self.final_data_filename = filename_stub + '_LSH_labels.csv'
        self.unfiltered_data_filename = filename_stub + '_full_LSH_labels.csv'

        self.data['LSH label'] = self.labels
        data_filtered = self.data.dropna(subset=[self.description])
        data_filtered.to_csv(self.unfiltered_data_filename, index=False)
        data_filtered = data_filtered[data_filtered['LSH label'] != -1]
        data_filtered.to_csv(self.final_data_filename, index=False)


    def clustering(self):
        ''' process each ad individually and incrementally save cluster graph. '''
        t = time.time()

        index = 0
        for _, row in self.data.iterrows():
            if index and not index % 10000:
                time_elapsed = time.time() - t
                print(index, '/', self.num_ads, 'time', time_elapsed)

            self.docid_to_index[row[self.id]] = index
            self.index_to_docid[index] = row[self.id]
            self.process_ad(index, row)
            index += 1

        self.generate_labels()
        self.write_cluster_graph()
        self.write_csv_labels()
        self.total_time = time.time() - t
        print('Finished clustering!', self.total_time)

        #print(self.num_ads_features, len(self.data.index))


    def get_clusters(self):
        ''' given cluster graph, return the relevant connected components '''
        criteria = lambda x: len(x) >= 5
        return [c for c in nx.connected_components(self.cluster_graph) if criteria(c)]


    def get_docs(self, cluster_nodes):
        ''' given a set of cluster nodes, return the documents they represent '''
        return cluster_nodes


    def print_clusters(self):
        print('number of clusters', len(self.get_clusters()))
        clusters = sorted(self.get_clusters(), key=lambda x: len(self.get_docs(x)), reverse=True)
        document_nodes = [n for n, d in self.cluster_graph.nodes(data=True) if d['bipartite']]
        for i, cluster in enumerate(clusters):
            docs = [c for c in cluster if c in document_nodes]
            print('cluster:', i, 'len:', len(docs))
            for doc_id in docs:
                index = self.docid_to_index[doc_id]
                row = self.data.loc[index]
                try:
                    description = row[self.description]
                except:
                    print('issue with doc_id', doc_id, 'and desc', self.description)
                print(doc_id, [n for n in self.cluster_graph.neighbors(doc_id)], row['label'])
                #print(doc_id, [n for n in self.cluster_graph.neighbors(doc_id)], row['is_spam'])
                print()
            print('\n\n')


def usage(exit_code):
    print('Usage: _ [filename]')
    exit(exit_code)


if __name__ == '__main__':
    # either just provide filename, or provide all params
    if len(sys.argv) not in [2, 5]:
        usage(1)

    filename = sys.argv[1]
    d = InfoShieldCoarse(filename, num_phrases=5)
    d.clustering()
