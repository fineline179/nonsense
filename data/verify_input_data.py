import os, csv, glob

direc = os.getcwd()

def verify_input_data(directory):
  os.chdir(directory)
  files = glob.glob('words_labeled*.csv')

  valid_labels = ['0', '1']
  for file in files:
    with open(os.path.join(directory, file), 'r') as f:
      lines = [row for row in csv.reader(f, delimiter=',')]

    file_valid = True
    for i, line in enumerate(lines):
      if line[1] in valid_labels:
        continue
      else:
        print(file + " is INVALID in line " + str(i + 1))
        file_valid = False
        break

    if file_valid:
      print(file + " is VALID!")


verify_input_data(direc)