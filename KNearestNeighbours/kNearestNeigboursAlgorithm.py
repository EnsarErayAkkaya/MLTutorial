import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))) # gets current directory

def k_nearest_neighbours(data, predict, k=3):
    if( len(data) >= k ):
        warnings.warn('K is set to a value less than total voting groups!')
    distances= []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm( np.array(features) - np.array(predict) )
            distances.append([euclidean_distance, group])
        
    votes = [ i[1] for i in sorted(distances)[:k] ]
    vote_result = Counter(votes).most_common(1)[0][0] # get first elemet of first elemt returned list of list, mostcommont return a tuple list as [(data, count)]
    confidence = Counter(votes).most_common(1)[0][1] / k # percent of most common element to other elements

    return vote_result, confidence

df = pd.read_csv(os.path.join(__location__, 'breast-cancer-wisconsin.txt')) # get file in directory
df.replace('?', -99999, inplace = True) # int his dataset all empty values marked by '?' we cahnged to -99999 for making them unused
df.drop(['id'], 1, inplace = True) # id is irrevelant for this example so we are droping it
full_data = df.astype(float).values.tolist() # some data can be string so make it float to be certain 
random.shuffle(full_data) # shuffle all data

test_size = 0.3
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[: -int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)) :]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbours(train_set, data, k=5)
        if vote == group :
            correct += 1
        total += 1
print('Accuracy:', correct/total )