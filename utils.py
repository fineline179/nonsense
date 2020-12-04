import csv
import numpy as np
import os
import glob
import csv

def read_csv(filename):
  word = []
  type = []

  with open(filename) as csvDataFile:
    csvReader = csv.reader(csvDataFile)

    for row in csvReader:
      word.append(row[0])
      type.append(row[1])

  X = np.asarray(word)
  Y = np.asarray(type, dtype=int)

  return X, Y

