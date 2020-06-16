#!/usr/bin/env python3
# Author:   Catalina Vajiac
# Purpose:  Use LSH idea to cluster text data
# Usage:    ./streaming_alg.py [filename]

import heapq
import math
import networkx as nx
import numpy as np
import os, sys
import pandas
import pickle
import random
import re
import time

from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from sklearn.metrics.cluster import adjusted_rand_score, homogeneity_score, normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer


class hash_family():
    def __init__(self, num_hash_functions, buckets_in_hash):
        self.num_hash_functions = num_hash_functions
        self.hash_cutoff = num_hash_functions / 2
        self.duplicate_cutoff = num_hash_functions * 3 / 4
        self.buckets_in_hash = buckets_in_hash

        random_generator = lambda: random.randrange(buckets_in_hash)
        self.hash_functions = [defaultdict(random_generator) for _ in range(num_hash_functions)]
        self.hash_tables = [defaultdict(list) for _ in range(num_hash_functions)]
        self.marked = set()
        self.max_bucket = 0
        self.max_comparisons = 0


    def get_hashes(self):
        ''' return pairs of hash functions and hash tables '''
        for h, table in zip(self.hash_functions, self.hash_tables):
            yield h, table

    def add_to_hash_tables(self, to_hash, to_add):
        ''' adds to_add in place that to_hash hashes, for all hash functions '''
        for h, table in zip(self.hash_functions, self.hash_tables):
            table[h[to_hash]] = list(set([val for val in table[h[to_hash]] if val not in self.marked]))
            if to_add not in table[h[to_hash]]:
                table[h[to_hash]].append(to_add)

            if len(table[h[to_hash]]) > 10:
                table[h[to_hash]].pop(0)

            self.max_bucket = max(self.max_bucket, len(table[h[to_hash]]))


    def __repr__(self):
        ''' prints all nonzero buckets, with elements, for each hash table '''
        for index, table in enumerate(self.hash_tables):
            print('Table', index)
            for bucket, elements in table.items():
                print('  ', bucket, ':', ' '.join(map(str, elements)))



class AutoDupCoarse():
    def __init__(self, filename, id_text='', description_text=''):
        self.filename_full = filename.split('.')[0]
        self.filename = os.path.basename(filename).split('.')[0]
        self.time_filename = self.filename + '_time.txt'
        self.ngrams = [5, 5]
        self.time = 0
        self.id_to_index = Counter()
        self.index_to_id = Counter()
        num_hash_functions=128

        self.data = pandas.read_csv(filename)
        columns = set(self.data.columns)

        # automatically determine relevant header names
        descriptions = {'u_Description', 'description', 'body', 'Tweet', 'text'}
        indices = {'ad_id', 'index', 'TweetID', 'id'}
        descriptions.add(description_text)
        indices.add(id_text)
        for name, field in [('description', descriptions), ('unique id', indices)]:
            if not len(columns.intersection(field)):
                print('Add "{}" header to possible descriptions!'.format(name))
                exit(1)
        self.description = set(self.data.columns).intersection(descriptions).pop()
        self.id = set(self.data.columns).intersection(indices).pop()

        if 'timestamp' in self.data.columns:
            self.data.sort_values(by=['timestamp']) # since ads not in order as they should be
        self.num_ads = len(self.data.index)
        self.cluster_graph = nx.DiGraph()
        self.hashes = hash_family(num_hash_functions, math.floor(len(self.data.index) / 100))


    def filter_text(self, text):
        ''' return preprocessed text, i.e. without digits and html '''
        if type(text) is not str: # nan
            return ''

        replace = [(r'\d+', ''), (r'[^\x00-\x7F]+', ' '), (re.compile('<.*?>'), '')]
        for source, target in replace:
            text = re.sub(source, target, text)

        return text


    def tokenize_text(self, text):
        return self.tokenizer(text)


    def find_tfidf(self):
        ''' pre-calculate tfidf '''
        print('Finding tfidf...')
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=self.ngrams, norm='l2',
                smooth_idf=True, min_df=2, max_df=0.8)
        self.data[self.description] = self.data[self.description].apply(self.filter_text)
        self.tfidf = vectorizer.fit_transform(self.data[self.description])
        self.tfidf_indices = vectorizer.get_feature_names()
        self.tokenizer = vectorizer.build_analyzer()


    def top_tfidf_phrases(self, index, return_all=False):
        ''' return the top phrases with highest tfidf score '''
        t = time.time()

        row, indices = self.tfidf[index].nonzero()
        tfidf_sorted = list(zip(self.tfidf[row, indices], indices))
        quarter_phrases = math.floor(len(indices) / 5)

        self.time += (time.time() - t)
        return heapq.nlargest(quarter_phrases, tfidf_sorted)
        return tfidf_sorted[:quarter_phrases]


    def find_related_clusters(self, phrases, doc_id):
        ''' return dict of related clusters, cluster type, and cluster id
            hash phrase for all k hash functions, find which clusters are related '''

        #related_clusters = defaultdict(lambda: {p: 0 for p in phrases})
        related_clusters = defaultdict(lambda: Counter())
        related_set = set()
        for phrase in phrases:
            for h, table in self.hashes.get_hashes():
                for cluster in table[h[phrase]]:
                    if cluster in self.hashes.marked:
                        continue
                    related_clusters[cluster][phrase] += 1

        for cluster, phrase_count in related_clusters.copy().items():
            phrases_related = [count >= self.hashes.hash_cutoff for _, count in phrase_count.items()]
            if all(phrases_related):
                return {}, 'duplicate', cluster

            if any(phrases_related):
                related_set.add(cluster)

        self.hashes.max_comparisons = max(len(related_clusters), self.hashes.max_comparisons)
        cluster_id = self.cluster_graph.number_of_nodes()
        self.hashes.marked.update(related_set)
        return related_set, 'chain', cluster_id


    def add_new_cluster(self, related_clusters, cluster_id, doc_id):
        ''' add cluster to cluster_graph, with related edges '''
        self.cluster_graph.add_node(cluster_id, contains=list([doc_id]))
        self.cluster_graph.add_edges_from([(rel, cluster_id) for rel in related_clusters])


    def process_ad(self, index, row):
        ''' find top phrases, use them to find related clusters, and add the ad to the cluster
        graph '''
        doc_id = row[self.id]

        top_tfidf = [phrase for _, phrase in self.top_tfidf_phrases(index)]
        related_clusters, cluster_type, cluster_id = self.find_related_clusters(top_tfidf, doc_id)

        if cluster_type == 'duplicate':
            self.cluster_graph.nodes[cluster_id]['contains'].append(doc_id)
            return

        self.add_new_cluster(related_clusters, cluster_id, doc_id)

        for phrase in top_tfidf:
            self.hashes.add_to_hash_tables(phrase, cluster_id)


    def write_cluster_graph(self):
        ''' write cluster graph as pkl file '''
        if not os.path.isdir('pkl_files'):
            os.mkdir('pkl_files')

        with open('pkl_files/{}_ad_graph.pkl'.format(self.filename), 'wb') as f:
            pickle.dump(self.cluster_graph, f)


    def write_csv_labels(self):
        ''' write new csv, with LSH labels '''
        self.final_data_filename = self.filename_full + '_LSH_labels.csv'

        my_labels = [-1]*len(self.data.index)
        for i, cluster in enumerate(self.get_clusters()):
            for ad in self.get_docs(cluster):
                my_labels[self.id_to_index[ad]] = i

        self.data['LSH label'] = my_labels
        data_filtered = self.data.dropna(subset=[self.description])
        is_keep = lambda x: len(self.tokenize_text(x)) >= 5
        data_filtered = data_filtered[data_filtered[self.description].map(is_keep)]
        data_filtered = data_filtered[data_filtered['LSH label'] != -1]
        data_filtered.to_csv(self.final_data_filename)


    def clustering(self):
        ''' process each ad individually and incrementally save cluster graph. '''
        t = time.time()
        self.find_tfidf()
        print('Finished tfidf in time:', time.time() - t)
        print('Starting clustering...')

        # assume in order of timestamp (streaming case)
        for index, row in self.data.iterrows():
            if index and not index % 10000:
                time_elapsed = time.time() - t
                print(index, '/', self.num_ads, 'time', time_elapsed)
                print('\t', self.time)
                print('\t', 'max bucket size', self.hashes.max_bucket, self.hashes.max_comparisons)
                #self.write_cluster_graph()

            self.id_to_index[row[self.id]] = index
            self.index_to_id[index] = row[self.id]
            self.process_ad(index, row)

        self.write_cluster_graph()
        self.write_csv_labels()
        self.total_time = time.time() - t
        print('Finished clustering!', self.total_time)
        with open('TIMES.txt', 'a') as f:
            f.write('{} {}\n'.format(self.filename, self.total_time))


    def visualize_buckets(self):
        ''' creates visual representation of how full the buckets are for a hash table '''
        print('Plotting hash tables...')
        save_path = './plots/streaming_alg/' + self.filename
        print(save_path)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        length = self.cluster_graph.number_of_nodes()
        for index in range(self.hashes.num_hash_functions):
            hash_table = self.hashes.hash_tables[index]
            m = np.zeros((len(hash_table), length))
            for i, (_, cluster_ids) in enumerate(hash_table.items()):
                m[i, cluster_ids] = 1

            plt.imshow(m, interpolation='nearest', aspect='auto')
            plt.tight_layout()
            plt.title('Hash table: {}'.format(index))
            plt.xlabel('cluster ids')
            plt.ylabel('buckets, sorted by first access time')
            plt.savefig('{}/{}_hash_visual.png'.format( save_path, index))
            plt.clf()


    def get_clusters(self):
        ''' given cluster graph, return the relevant connected components '''
        criteria = lambda x: len(x) > 1 or len(self.get_docs(x)) > 1
        return [c for c in nx.weakly_connected_components(self.cluster_graph) if criteria(c)]


    def get_docs(self, cluster_nodes):
        ''' given a set of cluster nodes, return the documents they represent '''
        return [ad for node in cluster_nodes for ad in self.cluster_graph.nodes[node]['contains']]


    def print_clusters(self):
        print('number of clusters', len(self.get_clusters()))
        clusters = sorted(self.get_clusters(), key=lambda x: len(self.get_docs(x)), reverse=True)
        for i, cluster in enumerate(clusters):
            print('cluster', i, 'len', len(cluster))
            for doc_id in self.get_docs(cluster):
                index = self.id_to_index[doc_id]
                row = self.data.loc[index]
                try:
                    description = row[self.description]
                except:
                    print('issue with doc_id', doc_id, 'and desc', self.description)
                #print(row['IsLegitimate'], type(row['IsLegitimate']))
                #true_label = row['UserID'] if not row['IsLegitimate'] else -1
                true_label = row['user_id']
                print(true_label, row['is_bot'], description)
                print()
            print('\n\n')


    def compare_true_synthetic(self):
        true_labels = self.data['user_id'].values
        is_bot= self.data['is_bot'].values

        true_labels = [label if flag else -1 for label, flag in zip(true_labels, is_bot)]

        my_labels = [-1]*len(self.data.index)
        for i, cluster in enumerate(self.get_clusters()):
            for ad in self.get_docs(cluster):
                index = self.id_to_index[ad]
                my_labels[index] = i

        nmi = normalized_mutual_info_score(true_labels, my_labels)
        hom = homogeneity_score(true_labels, my_labels)
        ari = adjusted_rand_score(true_labels, my_labels)
        print('NMI        ', nmi)
        print('HOMOGENEITY', hom)
        print('RAND INDEX ', ari)
        return nmi, hom, ari


def usage(exit_code):
    print('Usage: _ [filename]')
    exit(exit_code)


if __name__ == '__main__':
    # either just provide filename, or provide all params
    if len(sys.argv) not in [2, 5]:
        usage(1)

    filename = sys.argv[1]

    d = AutoDupCoarse(filename)
    d.clustering()
    #d.print_clusters()
    #score = d.compare_true_synthetic()
