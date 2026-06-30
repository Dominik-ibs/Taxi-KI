import pandas as pd
from sklearn.model_selection import train_test_split 

chunksize = 2_000_000

# Put the tripdata.csv next to this script before running it
filepath = 'tripdata.csv'

train_row_count = 0
test_row_count = 0

mode = 'w'

for chunk in pd.read_csv(filepath, chunksize=chunksize, low_memory=False):
    train_set, test_set = train_test_split(chunk, test_size=0.2, random_state=69)

    train_set.to_csv('train.csv', mode=mode, index=False, header=mode=='w')
    test_set.to_csv('test.csv', mode=mode, index=False, header=mode=='w')

    train_row_count += len(train_set)
    test_row_count += len(test_set)

    mode = 'a'

print(f'Train rows: {train_row_count}')
print(f'Test rows: {test_row_count}')