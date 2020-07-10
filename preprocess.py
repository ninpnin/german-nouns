import pickle
from numpy import genfromtxt
import re
import random

# Read and polish data
MAXLEN = 15

def polish_wd(wd):
    wd = wd.lower()
    wd = re.sub(r'[^a-zäöüß_]', '', wd).split("_")[-1]
    wd = re.sub(r'\d+', '', wd)
    wd = "_" * MAXLEN + wd
    return wd[-MAXLEN:]

def read_polish(name):
    dataset = genfromtxt("data/" + name + ".csv", delimiter=',', dtype=str)
    dataset = dataset[:,1] # Only take words from the table
    dataset = dataset[1:]
    dataset = list(dataset)
    dataset = [polish_wd(wd) for wd in dataset]
    dataset = list(set(dataset))
    print(dataset)
    return dataset

masculine = read_polish("masculine")
feminine = read_polish("feminine")
neuter = read_polish("neuter")

masculine = [(wd, "masculine") for wd in masculine]
feminine = [(wd, "feminine") for wd in feminine]
neuter = [(wd, "neuter") for wd in neuter]

# Split it into train, test, valid sets

mfn = masculine + feminine + neuter
random.shuffle(mfn)
print(mfn)
wd_count = len(mfn)

first = wd_count * 70 // 100
second = wd_count * 85 // 100

train = mfn[:first]
valid = mfn[first:second]
test = mfn[second:]

print("Length of train set", len(train))
print("Length of valid set", len(valid))
print("Length of test set", len(test))

# Pickle it!

pickle.dump(train, open("data/train.pkl", "wb"))
pickle.dump(valid, open("data/valid.pkl", "wb"))
pickle.dump(test, open("data/test.pkl", "wb"))
