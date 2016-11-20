import json
from pprint import pprint

with open('data.json') as data_file:
    data = json.load(data_file)

    lst = []
    for  i in data:
        lst = lst + data[i]['items']
    print len(set(lst))
