#%%
from itertools import product
from string import ascii_lowercase

#%%
def gen_letter_combinations(word_len=3):
  """Returns all words of length 'word_len'

  Args:
    word_len(int): length of word (default 3)

  Returns:
    List of strings of all letter combinations

  """

  return ["".join(x) for x in list(product(list(ascii_lowercase), repeat=word_len))]


words = gen_letter_combinations(3)


