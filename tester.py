import pickle

list = [[1,2], [3,4]]
with open('replay.pkl', 'wb') as w:
    pickle.dump(list, w)


with open('replay.pkl', 'rb') as r:
    array = pickle.load(r)
    print(array)
