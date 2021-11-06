# Generate random words and write to file

#%%
import csv
import os
from itertools import product
from string import ascii_lowercase
import random

#%%
# For generating letter combos to file
random.seed(1)

WORD_LEN = 4
OUTPUT_PATH = "/home/fineline/projects/nonsense/data"

# generate all 4 letter combinations
words = ["".join(x) for x in list(product(list(ascii_lowercase), repeat=WORD_LEN))]
random.shuffle(words)


def output_words(low, high):
  """ write out letter combinations in range (low, high) to file"""
  assert 0 <= low < high <= len(words)
  words_to_output = words[low:high]
  filename = f"words_{low}_{high}.csv"
  with open(os.path.join(OUTPUT_PATH, filename), "w") as f:
    for word in words_to_output:
      f.write(word + "\n")


#%%
# For generating letters combos and returning list
def return_words(low, high):
  """ return letter combinations in range (low, high)"""
  word_len = 4
  random.seed(1)
  # all 4 letter combinations
  all_words = [
    "".join(x) for x in list(product(list(ascii_lowercase), repeat=word_len))
  ]
  random.shuffle(all_words)
  assert 0 <= low < high <= len(all_words)
  return all_words[low:high]


#%%
output_words(10000, 15000)
