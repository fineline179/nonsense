# Generate random words and write to file

#%%
import csv
import os
from itertools import product
from string import ascii_lowercase
import random
random.seed(1)

WORD_LEN = 4
OUTPUT_PATH = '/home/fineline/projects/nonsense/data'

words = ["".join(x) for x in list(product(list(ascii_lowercase), repeat=WORD_LEN))]
random.shuffle(words)

#%%
words_to_output = words[10000:15000]

#%%
def output_words(filename, words):
  with open(os.path.join(OUTPUT_PATH, filename), 'w') as f:
    for word in words:
      f.write(word + '\n')


#%%
output_words('words_10000_15000.csv', words_to_output)


