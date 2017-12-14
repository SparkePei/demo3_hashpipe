#! /usr/bin/python
import socket
import time,numpy,struct
PKTSIZE = 4096
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = ("127.0.0.2",5009)
fp=open('file0000','r')
data=numpy.fromstring(fp.read(),dtype='b')
for i in range(0,len(data)/PKTSIZE):
	#sock.sendto(str(data[i*PKTSIZE:(i+1)*PKTSIZE]),addr)
	n=sock.sendto((data[i*PKTSIZE:(i+1)*PKTSIZE]),addr)
	print "send %d bytes of number %d packets to local address! "%(n,i)
	time.sleep(0.01) #0.000001 sec(100ns) no packets loss
fp.close()
