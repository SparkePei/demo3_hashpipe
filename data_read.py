#!/usr/bin/python

import numpy
f=open('demo3_file.txt','r')
data=numpy.fromstring(f.read(),dtype='<i4')
for i in range(0,len(data)):
	print "number %d sum is:%d"%(i+1,data[i])
