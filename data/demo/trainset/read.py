# -*- coding:utf8 -*-
import json 
import sys

f = open(sys.argv[1])
data = f.readline()
data = json.loads(data)

print len(data)
print data.keys()
