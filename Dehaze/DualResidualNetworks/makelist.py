import os

path = 'data/test/labels'
names = os.listdir(path)
f = open('data/test/list', 'a')
for name in names:
    print(name, file=f)
f.close()