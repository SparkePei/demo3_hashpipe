#! /usr/bin/python
# Import required modules
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, lfilter
import seaborn as sns
import socket
import time,numpy,struct

#sns.set_style("white")
def db(x):
    """ Convert linear value to dB value """
    x+=0.0000001
    return 10*np.log10(x)
def pfb_fir_frontend(x, win_coeffs, M, P):
    W = x.shape[0] / M / P
    x_p = x.reshape((W*M, P)).T
    h_p = win_coeffs.reshape((M, P)).T
    x_summed = np.zeros((P, M * W - M))
    for t in range(0, M*W-M):
        x_weighted = x_p[:, t:t+M] * h_p
        x_summed[:, t] = x_weighted.sum(axis=1)
    return x_summed.T
def generate_win_coeffs(M, P, window_fn="hamming"):
    win_coeffs = scipy.signal.get_window(window_fn, M*P)
    sinc       = scipy.signal.firwin(M * P, cutoff=1.0/P, window="rectangular")
    #sinc       = scipy.signal.firwin(M * P, cutoff=1.0/P, window="hamming")
    win_coeffs *= sinc
    return win_coeffs
def fft(x_p, P, axis=1):
    return np.fft.fft(x_p, P, axis=axis)
def correlator(x,y,n_taps, n_chan, n_int, window_fn="hamming"):
    M = n_taps
    P = n_chan 
    
    # Generate window coefficients
    win_coeffs = generate_win_coeffs(M, P, window_fn)  
    
    # Apply frontend, take FFT
    x_pfb = pfb_fir_frontend(x, win_coeffs, M, P)
    x_fft = fft(x_pfb, P)
    y_pfb = pfb_fir_frontend(y, win_coeffs, M, P)
    y_fft = fft(y_pfb, P)
    
    # Perform cross correlation
    xx = x_fft.real**2    
    yy = y_fft.real**2
    xy_real =  x_fft.real * y_fft.real
    xy_imag = x_fft.imag * y_fft.imag

    # Trim array so we can do time integration
    xx = xx[:np.round(xx.shape[0]//n_int)*n_int]
    yy = yy[:np.round(yy.shape[0]//n_int)*n_int]
    xy_real = xy_real[:np.round(xy_real.shape[0]//n_int)*n_int]
    xy_imag = xy_imag[:np.round(xy_imag.shape[0]//n_int)*n_int]

    # Integrate over time, by reshaping and summing over axis (efficient)
    xx = xx.reshape(xx.shape[0]//n_int, n_int, xx.shape[1])
    xx = xx.mean(axis=1)
    
    yy = yy.reshape(yy.shape[0]//n_int, n_int, yy.shape[1])
    yy = yy.mean(axis=1)
    
    xy_real = xy_real.reshape(xy_real.shape[0]//n_int, n_int, xy_real.shape[1])
    xy_real = xy_real.mean(axis=1)
    
    xy_imag = xy_imag.reshape(xy_imag.shape[0]//n_int, n_int, xy_imag.shape[1])
    xy_imag = xy_imag.mean(axis=1)
    
    return xx,yy,xy_real,xy_imag
if __name__ == "__main__":

    #M     = 4          # Number of taps
    M     = 1          # Number of taps
    #P     = 1024       # Number of 'branches', also fft length
    P     = 4096       # Number of 'branches', also fft length
    W     = 10       # Number of windows of length M*P in input time stream
    #n_int = 2          # Number of time integrations on output data
    
    samples = np.arange(-1.0,1.0,2.0/P)
    
    noise1   = np.random.random(M*P*W) 
    noise2   = np.random.random(M*P*W) 
    freq = 500
    
    #amp  = 0.02
    amp  = 0.2
    
    phase1 = np.pi/4
    phase2 = np.pi/2
    
    cw_signal1 = M*W*(amp*np.sin(2*np.pi*samples*freq+phase1)).tolist()
    cw_signal2 = M*W*(amp*np.sin(2*np.pi*samples*freq+phase2)).tolist()
    
    data1 = noise1 + cw_signal1
    data2 = noise2 + cw_signal2
    PKTSIZE = 4096
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = ("127.0.0.1",5009)
    for i in range(0,len(data1)/PKTSIZE):
        #sock.sendto(str(data[i*PKTSIZE:(i+1)*PKTSIZE]),addr)
        n=sock.sendto((data1[i*PKTSIZE:(i+1)*PKTSIZE]),addr)
        n=sock.sendto((data2[i*PKTSIZE:(i+1)*PKTSIZE]),addr)
        print "send %d bytes of number %d packets to local address! "%(n,i)
        time.sleep(0.001) #0.000001 sec(100ns) no packets loss
    
    '''
    plt.subplot(2,1,1)
    plt.title("Sin wave")
    plt.plot(samples[:1024]*360*freq,cw_signal1[:1024],label='Sin wave 1')
    plt.plot(samples[:1024]*360*freq,cw_signal2[:1024],label='Sin wave 2')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
    plt.legend()
    plt.subplot(2,1,2)
    plt.title("Noise + sin",fontsize=16)
    plt.plot(data1[:1024],label='Noise + sin 1')
    plt.plot(data2[:1024],label='Noise + sin 2')
    plt.legend()
    '''
    XX,YY,XY_REAL,XY_IMAG = correlator(data1,data2, n_taps=M, n_chan=P, n_int=1000, window_fn="hamming")
    
    plt.subplot(3,1,1)
    plt.plot(db(XX[0]), label='XX')
    plt.plot(db(YY[0]), label='YY')
    plt.ylim(-50, -32)
    plt.xlim(0, P/2)
    plt.xlabel("Channel",fontsize=16)
    plt.ylabel("Power [dB]")
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.plot(db(XY_REAL[0]), label='XY_REAL')
    plt.plot(db(XY_IMAG[0]), label='XY_IMAG')
    plt.ylim(-90, -20)
    plt.xlim(0, P/2)
    plt.xlabel("Channel",fontsize=16)
    plt.ylabel("Power [dB]")
    plt.legend()
    
    phase = len(XY_REAL[0])*[0]
    for i in range(len(XY_REAL[0])):
        phase[i] = np.angle(complex(XY_REAL[0][i],XY_IMAG[0][i]))/np.pi*180.0
        
    plt.subplot(3,1,3)
    plt.plot(phase,label='phase')
    plt.xlim(0, P/2)
    plt.xlabel("Channel",fontsize=16)
    plt.ylabel("Phase [degree]")
    plt.legend()
    plt.show()
