#Documents/Coding_all/NYCU

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plt_ticker
from pylab import *
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.signal import find_peaks
import pandas as pd
from IPython.display import display, Math, Latex



def read_file(file_name):
    
    '''
    file name: 'total-08-00-00-01011904.txt'
    '''
    rawdat = np.loadtxt(file_name, delimiter='\t')  

    xdat = rawdat[:,0]
    ydat = rawdat[:,1]
    
    return xdat, ydat
# read_file('total-08-00-00-01011904.txt')

def myerf(x, a, k, x0, y0):
    
    '''
    2a: delta of f(-inf) to f(inf)
    k : slope
    x0: mid of func
    c : mid of y
    '''
    
    return a * erf(k * (x - x0)) + y0

def gauss(x, μ, σ, A, y0):            
    return y0 + A * np.exp(-(x-μ)**2/2/σ**2) 

def Differential_array(x, y):

    dy = []
    for i, val in enumerate(y):
        if i != 0:
            a = (val - y[i-1])/(x[i] - x[i-1])
            dy.append(a)
    x = np.delete(x, 0)
    return x, dy


def plt_xy(x, y, A):
    fig = plt.figure(figsize=(6, 4), dpi=100)
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y, 'b.',markersize = 4)
    plt.show()
    if A == '.':
        ax1.plot(x, y, 'b.',markersize = 4)
    elif A == '.-':
        ax1.plot(x, y, 'b.-')

def fit_error(xdat, ydat, mid_x):
    
    '''
    2a: delta of f(-inf) to f(inf)
    k : slope
    x0: mid of y 對應的 x (func 中心)
    c : mid of y
    '''
    p0=[0.023, 80.0, mid_x, data_list[0]]
    popt, pcov = curve_fit(myerf, xdat, ydat, p0, method="lm")
    perr = np.sqrt(np.diag(pcov)) * 4.0
#     σ = np.sqrt(np.diag(pcov))        
#     a = pd.DataFrame(data={'params':popt,'σ':σ}, index = myerf.__code__.co_varnames[1:])      
#     display(a)   

    return popt

def fit_gauss():
    
    '''
    微分後 fit
    '''
    x_new, y_new, data_list = skip_noise()
    x, dy = Differential_array(x_new, y_new)

    mid_ofx, xpeaks = Find_thepeak(x_new, y_new, 3) # 可刪

    # p0=[0.023, 80.0, mid_x, data_list[0]]
    # popt, pcov = curve_fit(gauss, xdat, ydat, p0, method="lm")
    # perr = np.sqrt(np.diag(pcov)) * 4.0
    # σ = np.sqrt(np.diag(pcov))        
    # a = pd.DataFrame(data={'params':popt,'σ':σ}, index = myerf.__code__.co_varnames[1:])      
    # display(a)  

    fig = plt.figure(figsize=(6, 4), dpi=100)
    ax1 = fig.add_subplot(111)
    ax1.plot(x, dy, 'b.',markersize = 4)
    x_c = float(mid_ofx[0])
    ax1.set_xlim(x_c-0.1, x_c+0.1)
    # ax1.plot(x, y, 'b.',markersize = 4)
    plt.show()

    return 

def skip_noise():
    
    '''
    First:
    Find the noisce data. Useing histogram to find noise.
    
    Second:
    Using noise range (noise_bottom ~ noise_top) skip the noise data.
    '''
    # load file
    # xdat, ydat = read_file('100--1-21-05-57-11212022.txt')
    xdat, ydat = read_file("景貴刀口微分.txt") # 刪除
    plt_xy(xdat, ydat, '.')

    # First step:
    
    counts, dis = np.histogram(ydat, bins= 10)    
#     plt.stairs(counts, dis) # 可刪
    _y = counts.tolist()
    max_index = _y.index(max(counts)) 
    delta = float(format(dis[1] - dis[0], '.7f'))
    # print( dis[max_index] + delta/2 ) # 可刪
    
    noise_top = dis[max_index] + delta/2
    noise_bottom = dis[max_index] - delta/2
    
    '''
    Use distribution 
    first : noise
    second: up or bottem data
    third : up or bottem data
    '''
    
    max1 = counts.tolist().index(sorted(counts, reverse = True)[1])
    max2 = counts.tolist().index(sorted(counts, reverse = True)[2])
    mid_y = (dis[max1] + dis[max2]) / 2
    
    # define the up data adn below data
    up_data, down_data, data_list = float(0), float(0), []
    if dis[max2] > dis[max1]:
        up_data = dis[max2]
        down_data = dis[max1]
    else:
        up_data = dis[max1]
        down_data = dis[max2]
    data_list = [mid_y, up_data, down_data]

    
    # Second step:

    '''
    noise might be two line.
    i need to skip all of it.
    '''
      
    xx, yy = [], []
    yyy = sorted(  ydat.tolist()  )
    
    for i, val_y in enumerate(ydat):
        if val_y > yyy[counts[0]]:
            xx.append(xdat[i])
            yy.append(ydat[i]) 
#     plt_xy(xx, yy, '.')  # noise repair point.
    return xx, yy, data_list  
    '''
    x_new, y_new 代表去除雜訊後的 date
    data_list 用在 fittint error mid
    '''
x_new, y_new, data_list = skip_noise()

def gradient(x, y, N):
    
    '''
    average every N points.

    Gral: find the fitting point. 
    return the mid of error func x position.
    
    First, the data is averaged every N points.
    Then, useing gradient method find the throld point.
    '''

    ave_x, ave_y, y_grad, y_grad_ = [], [], [], []
    for i in range(0,len(y)- N, N):
        total = 0.0
        i = int(i)
        
        for j in range(N-1):
            total += y[i+j]
        ave_y.append(total / N)
        if i == 0:
            continue   # pass first, beacause of gradient.
        ave_x.append(x[i])


    # gradient data
    for i in range(len(ave_y)-2):
        y = (ave_y[i+2] - ave_y[i]) / 2    
        y_grad.append(y)
        y_grad_.append(abs(y))
    ave_x = ave_x[0:len(ave_x)-1]
    
    # plt_xy(ave_x ,y_grad_, '.-') # peaks repair point.

    return ave_x ,y_grad, y_grad_

def Find_thepeak(x_new, y_new, N):
    
    ave_x, y_grad, y_grad_= gradient(x_new, y_new, N)
    
    counts, dis = np.histogram(y_grad, bins= 5)
    delta = abs(dis[0] - dis[2])

    peaks, _ = find_peaks(y_grad_, height=delta)
    
    # '''
    # find peak 找到鄰近的值，需要移除
    # '''
    if len(peaks) != 1:
        for i, val in enumerate(peaks):
            if i == 0:
                pass
            elif abs(val - peaks[i-1]) < N*20 :
                peaks = np.delete(peaks, i)

    x, y = [], []   
    for i in peaks:
        x.append(x_new[i*N+3])   
        y.append(y_new[i*N+3])
        
    x_period_pixel = None
    if len(peaks) == 0:
        print('Error at find peaks! There is no peaks')
    elif len(peaks) == 1:
        pass
    elif len(peaks) == 2:
        x_period_pixel  = peaks[1] - peaks[0]
    else:
        x_period_1  = peaks[1] - peaks[0]
        x_period_2  = peaks[2] - peaks[1]
        if x_period_1 > x_period_2:
            x_period_pixel = x_period_2
        else:
            x_period_pixel = x_period_1      
    
    peak_end = peaks[len(peaks)-1]
    if x_period_pixel != None:
        if len(y_grad) - peak_end < x_period_pixel/10:
            np.delete(peaks, len(peaks)-1)
        if peaks[0] < x_period_pixel/10:
            np.delete(peaks, 0)


    # print(peaks)                   # peaks repair point.
    return x, peaks

def data_split_and_fit(N):
    
    mid_ofx, xpeaks = Find_thepeak(x_new, y_new, N)
    
    D_1, D_2 = [], []
    if len(xpeaks) == 1:
        '''
        D_1 = x_new 
        '''
        pass  
    else:
        for i, xval in enumerate(xpeaks):
            if i == 0:
                peak_mid = int( (xpeaks[i]+xpeaks[i+1])/2 )
                x = x_new[0:peak_mid*N]
                y = y_new[0:peak_mid*N]
                D_1.append(x)
                D_2.append(y)
            elif i != len(xpeaks)-1:
                peak_mid1 = int( (xpeaks[i-1]+xpeaks[i])/2 )
                peak_mid2 = int( (xpeaks[i]+xpeaks[i+1])/2 )
                x = x_new[peak_mid1*N:peak_mid2*N]
                y = y_new[peak_mid1*N:peak_mid2*N]
                D_1.append(x)
                D_2.append(y)
            elif i == len(xpeaks)-1:
                peak_mid = int( (xpeaks[i-1]+xpeaks[i])/2 )
                x = x_new[peak_mid*N:]
                y = y_new[peak_mid*N:]
                D_1.append(x)
                D_2.append(y)
    
    return D_1, D_2, mid_ofx
# D_1, D_2, mid_ofx = data_split_and_fit(3)

def plt_error_fit():
    
    D_1, D_2, mid_ofx = data_split_and_fit(3)

    fig = plt.figure(figsize=(10, 4), dpi=100)
    ax1 = fig.add_subplot(111)
    ax1.plot(x_new, y_new, 'b.',markersize = 5)
    ax1.set_xlabel("time (ms)")
    ax1.set_ylabel("signal (a.u.)")
    
    c = ['#B22222', '#CD9B1D', '#FF7D40', '#FFC125', '#FF3030', '#FFC125'] # color code for fit diff curve
    if len(mid_ofx) == 1:
            p = fit_error(x_new, y_new, mid_ofx[0])
            a, k, x0, y0 = p[0], p[1], p[2], p[3]
            ax1.plot(x_new, myerf(x_new, a, k, x0, y0), color=c[0] , linewidth=5, alpha=0.5)
    else:    
        for i, val in enumerate(mid_ofx):
            p = fit_error(D_1[i], D_2[i], val)

            a, k, x0, y0 = p[0], p[1], p[2], p[3]
            ax1.plot(D_1[i], myerf(D_1[i], a, k, x0, y0), color=c[i] , linewidth=5, alpha=0.5)
            # F.append(p)
            # print(a, k, x0, y0)
    plt.show()

def plt_gauss_fit():
    
    D_1, D_2, mid_ofx = data_split_and_fit(3)

    fig = plt.figure(figsize=(10, 4), dpi=100)
    ax1 = fig.add_subplot(111)
    ax1.plot(x_new, y_new, 'b.',markersize = 5)
    ax1.set_xlabel("time (ms)")
    ax1.set_ylabel("signal (a.u.)")
    
    c = ['#B22222', '#CD9B1D', '#FF7D40', '#FFC125', '#FF3030', '#FFC125'] # color code for fit diff curve
    if len(mid_ofx) == 1:
        p = fit_error(x_new, y_new, mid_ofx[0])
        a, k, x0, y0 = p[0], p[1], p[2], p[3]
        ax1.plot(x_new, myerf(x_new, a, k, x0, y0), color=c[0] , linewidth=5, alpha=0.5)
    else:    
        for i, val in enumerate(mid_ofx):
            p = fit_error(D_1[i], D_2[i], val)

            a, k, x0, y0 = p[0], p[1], p[2], p[3]
            ax1.plot(D_1[i], myerf(D_1[i], a, k, x0, y0), color=c[i] , linewidth=5, alpha=0.5)
            # F.append(p)
            # print(a, k, x0, y0)
    plt.show()

# plt_error_fit()
# fit_gauss()


# Differential_array(x_new, y_new)



