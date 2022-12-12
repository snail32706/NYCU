import glob, os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from IPython.display import display, Math, Latex
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# ------ --------- ------ #
# ------ Load File ------ #

def load_path():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    return folder_path

def r_all_file():
    '''
    get all .txt file.
    then write in a list: [file name, s=?, z=?, ...]
    '''
    folder_path = load_path()

    list_of_file = []
    # os.chdir("/Users/")
    os.chdir(folder_path)
    
    for file in glob.glob("*.txt"):
        if judgment_format(file) is not None:
            list_of_file.append(  judgment_format(file)  )
    '''
    list_of_file: [ ('file_name', speed, z),
                    ('file_name', speed, z),
                    ('file_name', speed, z) ]
    folder_path : 資料夾路徑
    '''
    return list_of_file, folder_path 

def judgment_format(file):
    '''
    split flie name by '_'
    '''
    file_name_all = str(file)
    if file_name_all[0] == 's':
        SS, ZZ, _ = file_name_all.split('_')
        _ , speed = SS.split('s')
        _ , z_component = ZZ.split('z')
        return file_name_all, int(speed), int(z_component)


def read_file(file_name):
    rawdat = np.loadtxt(file_name, delimiter='\t')  
    xdat = rawdat[:,0]
    ydat = rawdat[:,1]
    return xdat, ydat


# ------ -------- ------ #
# ------ def func ------ #

def myerf(x, a, k, x0, y0):
    '''
    2a: delta of f(-inf) to f(inf)
    k : slope
    x0: mid of func
    c : mid of y
    '''
    return a * erf(k * (x - x0)) + y0

def plt_xy(x, y, A, axis=None):
    fig = plt.figure(figsize=(6, 4), dpi=100)
    ax1 = fig.add_subplot(111)
    if axis is not None:
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("signal (a.u.)")
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
    p0=[0.023, 80.0, mid_x, mid_y]
    popt, pcov = curve_fit(myerf, xdat, ydat, p0, method="lm")
    perr = np.sqrt(np.diag(pcov)) * 4.0
    σ = np.sqrt(np.diag(pcov))        
    a = pd.DataFrame(data={'params':popt,'σ':σ}, index = myerf.__code__.co_varnames[1:])      
    # display(a)   
    return popt
def calculate_spot_size(k_parameters, speed):
    '''
    put all k
    calculate spot size and StD.
    '''
    spot_size_list = [] # unit: um^2
    for i, val in enumerate(k_parameters):
        k_um = val / speed 
        spot_size = 2*np.pi / ( k_um**2 ) # spot size = pi * r^2
        spot_size_list.append(spot_size)
    return spot_size_list 

# ------ ------------ ------ #
# ------ Data Process ------ #

def skip_noise(xdat, ydat):
    
    '''
    First:
    Find the noisce data. Useing histogram to find noise.
    
    Second:
    Using noise range (noise_bottom ~ noise_top) skip the noise data.
    '''

    # First step:
    counts, dis = np.histogram(ydat, bins= 10)    
    _y = counts.tolist()
    max_index = _y.index(max(counts)) 
    delta = float(format(dis[1] - dis[0], '.7f'))
    
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
    return xx, yy, mid_y 

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
    ave_x = ave_x[0:len(ave_x)-1]

    # gradient data
    for i in range(len(ave_y)-2):
        y = (ave_y[i+2] - ave_y[i]) / 2    
        y_grad.append(y)

    avg_x_byN = np.array(ave_x) # list 不方便計算
    del_y_byN = np.array(y_grad)
    return avg_x_byN, del_y_byN, N

def Find_thepeak(x_new, y_new):

    N = gradient(x_new, y_new, N)[2]
    counts, dis = np.histogram(del_y_byN, bins= 5)
    delta = abs(dis[0] - dis[2])
    peaks, _ = find_peaks(abs(del_y_byN), height=delta)
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
        if len(del_y_byN) - peak_end < x_period_pixel/10:
            np.delete(peaks, len(peaks)-1)
        if peaks[0] < x_period_pixel/10:
            np.delete(peaks, 0)
    return x, peaks

def data_split_and_fit(x_new, y_new):
    '''
    D_1, D_2 分別裝切割後的 diff error func region of data.
    N : avg of N number data.
    '''
    D_1, D_2 = [], []
    if len(xpeaks) == 1:
        D_1 = x_new
        D_2 = y_new
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
    return D_1, D_2


def append_all_k(D_1, D_2, mid_ofx):
    '''
    fit error func and append all k.
    '''
    k_parameter = []
    if len(mid_ofx) == 0:
        print('ERROR mid_ofx is None')
    elif len(mid_ofx) == 1:
        p = fit_error(x_new, y_new, mid_ofx[0])
        k_parameter.append(p[1])
    else:    
        for i, val in enumerate(mid_ofx):
            p = fit_error(D_1[i], D_2[i], val)
            k_parameter.append(p[1])
    return k_parameter

def create_z_list(list_of_file):
    '''
    list_of_file: [ ('file_name', speed, z),
                    ('file_name', speed, z),
                    ('file_name', speed, z) ]
    '''
    z_list = []
    for i in list_of_file:
        if i[2] not in z_list:
            z_list.append(i[2])
    return sorted(z_list, reverse = True) # z list 由大到小排列

def main():
    spot_dependence_on_z = [] # 總檔案 list

    list_of_file, folder_path = r_all_file()
    z_list = create_z_list(list_of_file) # 由大到小
    '''
    z 從最大值 開始讀檔
    '''
    for i, ZZ in enumeate(z_list):
        for f_list in list_of_file:  
            if i[2] == ZZ:
                '''
                z the same. Begine 
                '''
                absolute_file_path = f'{folder_path}/{f_list[0]}' # 絕對路徑

                xdat_r, ydat_r     = read_file(absolute_file_path)
                speed, z_component = f_list[1], f_list[2]

                x_new, y_new, mid_y     = skip_noise(xdat_r, ydat_r)
                avg_x_byN, del_y_byN, N = gradient(x_new, y_new, N)
                mid_ofx, xpeaks         = Find_thepeak(x_new, y_new)
                D_1, D_2                = data_split_and_fit(x_new, y_new) 

                k_parameters   = append_all_k(D_1, D_2, mid_ofx)
                spot_size_list = calculate_spot_size(k_parameters, speed)


            # fit error then calculate
 






main()






