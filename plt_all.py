import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import glob, os
import platform
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from IPython.display import display, Math, Latex
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import math

# ------ --------- ------ #
# ------ Load File ------ #

def open_file(): 
    '''
    使用相對路徑
    *** load data is the first step. ***
    use tkinter
    '''

    filetypes = ( ('text files', '*.txt'), ('All files', '*.*') )
    # global file_path_bytk
    file = filedialog.askopenfile(
                            title='Open a file',
                            initialdir='/Users/k.y.chen/Desktop/', 
                            filetypes=filetypes)
    file_path_bytk = str(file)
    # file type is <class '_io.TextIOWrapper'> 
    if file:
        file_path_bytk = os.path.abspath(file.name)
        # file_path_bytk = str(file_path_bytk)
    
    return file_path_bytk

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
            list_of_file.append(  judgment_format(file)[0]  )
    '''
    list_of_file: [ 'file_name', 'file_name', 'file_name', ... ]
    '''
    return list_of_file, folder_path 
    
def load_path():
    # win_l= tk.Tk()
    # win_l.withdraw()
    folder_path = filedialog.askdirectory()
    return folder_path

def judgment_format(file):
    '''
    split flie name by '_'
    [ file_name, speed, z ]  >> type: str, int, int
    '''
    file_name = str(file)
    if file_name[0] == 's':
        SS, ZZ, _ = file_name.split('_')
        _ , speed = SS.split('s')
        _ , z_component = ZZ.split('z')
        return file_name, int(speed), int(z_component)


def absolute_to_relative(absolute_file_path):

    if platform.system() == 'Darwin':
        file_name = str(absolute_file_path)
        for i in file_name.split('/'):
            if 'txt' in i:
                return i
                
    elif platform.system() == 'Windows':
        file_name = str(absolute_file_path)
        for i in file_name.split('\\'):
            if 'txt' in i:
                return i

def read_file(file_name):
    '''
    使用絕對路徑
    '''
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

def fit_error(xdat, ydat, mid_x, mid_y):
    
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
    parameter k is slope which mean k = Δy / Δx.
    If Δx = time(s) * speed(um/s). 
    Slope k(1/um) = k(1/s) * (1/speed)
    '''
    spot_size_list = [] # unit: um^2
    for i, val in enumerate(k_parameters):
        k_um = val / speed 
        spot_size = 2*np.pi / ( k_um**2 ) # spot size = pi * r^2
        spot_size_list.append(spot_size)
    return spot_size_list 

# ------ ------------ ------ #
# ------ Data Process ------ #

def find_midy(xdat, ydat):
    '''
    Use distribution 
    first : up or bottem data
    second: up or bottem data
    '''
    counts, dis = np.histogram(ydat, bins= 5)  
    # plt.stairs(counts, dis) # 可刪
    max1 = counts.tolist().index(sorted(counts, reverse = True)[0])
    max2 = counts.tolist().index(sorted(counts, reverse = True)[1])
    mid_y = (dis[max1] + dis[max2]) / 2
    return mid_y

def skip_noise(xdat, ydat):
    # First step:
    counts, dis = np.histogram(ydat, bins= 15)
     # plt.stairs(counts, dis) # 可刪
    region = []
    for i, val in enumerate(counts):
        if val > 50:
            region.append(dis[i])
            region.append(dis[i+1])
    N = int(len(region)/2)
    s_x, s_y = [], []
    for i in range(N):
        s_x.append([])
        s_y.append([]) # f'region{i+1}:'
    
    for i, val in enumerate(ydat):
        for j in range(N):
            if j == 0:
                if val < region[2*j+1]:
                    s_x[j].append(xdat[i])
                    s_y[j].append(ydat[i]) 
            elif j == N-1:
                if val > region[2*j]:
                    s_x[j].append(xdat[i])
                    s_y[j].append(ydat[i]) 
            else:
                if val > region[2*j] and val < region[2*j+1]:
                    s_x[j].append(xdat[i])
                    s_y[j].append(ydat[i]) 
    noise = []
    for j in range(N):
        delta = 0
        for i, val in enumerate(s_x[j]):
            if i == 0:
                delta = abs(s_x[j][i] - xdat[0]) if abs(s_x[j][i] - min(xdat)) > abs(s_x[j][len(s_x[j])-1] - xdat[len(xdat)-1]) else abs(s_x[j][len(s_x[j])-1] - xdat[len(xdat)-1])
            else:    
                delta = abs(s_x[j][i] - s_x[j][i-1]) if abs(s_x[j][i] - s_x[j][i-1]) > delta else delta
        if delta < (max(xdat) - min(xdat)) / 100:
            if len(noise) == 0:
                noise = s_x[j]
            else:
                noise += s_x[j]
                
    x_new, y_new = [], []       
    for i, val in enumerate(xdat):  
        if val not in noise:
            x_new.append(val)
            y_new.append(ydat[i])
    return x_new, y_new

def remove_top_bottom_noise(x_dat, y_dat):
    '''
    x_dat, y_dat is list. Result by "append"
    '''
    counts, dis = np.histogram(y_dat, bins= 5)  
    # plt.stairs(counts, dis) # 可刪

    if counts[0] < counts[1]:
        for i in range(counts[0]):
            idx = y_dat.index(min(y_dat))
            x_dat.pop(idx) 
            y_dat.pop(idx) 
    elif counts[4] < counts[3]:
#         print('T')
        for i in range(counts[4]):
            idx = y_dat.index(max(y_dat))
            x_dat.pop(idx) 
            y_dat.pop(idx) 
    elif counts[0] > counts[1] and counts[4] > counts[3]:
        break
    # plt_xy(x_dat, y_dat, '.')
    return x_dat, y_dat

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

def Find_thepeak(x_dat, y_dat, N):

    avg_x_byN, del_y_byN = gradient(x_dat, y_dat, N)[0], gradient(x_dat, y_dat, N)[1]
    counts, dis = np.histogram(del_y_byN, bins= 5)
    delta = abs(dis[0] - dis[2])
    peaks, _ = find_peaks(abs(del_y_byN), height=delta)
    '''
    peaks 鄰近值比大小
    '''
    peaks_list = peaks.tolist()
    if len(peaks) > 1:
        for i, val in enumerate(peaks):
            if i == 0:
                pass
            elif abs(val - peaks[i-1]) < N*20 :
                peaks_list.remove(  min(abs(val), abs(peaks[i-1]))   )
    peaks = np.array(peaks_list)

    x, y = [], []   
    for i in peaks:
        x.append(x_dat[i*N+3])   
        y.append(y_dat[i*N+3])
    
    x_period_pixel = None
    if len(peaks) == 0:
        print('Error at find peaks! There is no peaks')
    else:
        peak_end = peaks[len(peaks)-1]
        
        if len(peaks) == 1:
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

    if x_period_pixel != None:
        if len(del_y_byN) - peak_end < x_period_pixel/10:
            np.delete(peaks, len(peaks)-1)
        if peaks[0] < x_period_pixel/10:
            np.delete(peaks, 0)
            
    return x, peaks

def data_split_and_fit(x_dat, y_dat, xpeaks, N):
    '''
    D_1, D_2 分別裝切割後的 diff error func region of data.
    N : avg of N number data.
    '''
    D_1, D_2 = [], []
    if len(xpeaks) == 1:
        D_1 = x_dat
        D_2 = y_dat
    else:
        for i, xval in enumerate(xpeaks):
            if i == 0:
                peak_mid = int( (xpeaks[i]+xpeaks[i+1])/2 )
                x = x_dat[0:peak_mid*N]
                y = y_dat[0:peak_mid*N]
                D_1.append(x)
                D_2.append(y)
            elif i != len(xpeaks)-1:
                peak_mid1 = int( (xpeaks[i-1]+xpeaks[i])/2 )
                peak_mid2 = int( (xpeaks[i]+xpeaks[i+1])/2 )
                x = x_dat[peak_mid1*N:peak_mid2*N]
                y = y_dat[peak_mid1*N:peak_mid2*N]
                D_1.append(x)
                D_2.append(y)
            elif i == len(xpeaks)-1:
                peak_mid = int( (xpeaks[i-1]+xpeaks[i])/2 )
                x = x_dat[peak_mid*N:]
                y = y_dat[peak_mid*N:]
                D_1.append(x)
                D_2.append(y)
    return D_1, D_2

def append_all_k(D_1, D_2, mid_ofx, mid_y):
    '''
    fit error func and append all k.
    '''
    k_parameter = []
    if len(mid_ofx) == 0:
        print('ERROR mid_ofx is None')
    elif len(mid_ofx) == 1:
        p = fit_error(D_1, D_2, mid_ofx[0], mid_y)
        k_parameter.append(p[1])
    else:    
        for i, val in enumerate(mid_ofx):
            p = fit_error(D_1[i], D_2[i], val, mid_y)
            k_parameter.append(p[1])
    return k_parameter

def create_z_list(list_of_file):
    '''
    list_of_file: [ 'file_name', 'file_name', 'file_name', ... ]
    '''
    z_list = []
    for i in list_of_file:
        if judgment_format(i)[2] not in z_list:
            z_list.append( judgment_format(i)[2]
                                                )
    return sorted(z_list, reverse = True) # z list 由大到小排列

def main():

    '''
    Open folder then load all .txt file
    '''

    spot_dependence_on_z = [] # 總檔案 list
    chack_file_list, proplem_list = [], []

    list_of_file, folder_path = r_all_file()
    z_list = create_z_list(list_of_file) # 由大到小
    '''
    z 從最大值 開始讀檔
    '''
    for i, ZZ in enumerate(z_list):
        spot_dependence_on_z.append([ZZ])
        chack_file_list.append([ZZ])

        for f_name in list_of_file:  
            
            file_name, speed, z_int = judgment_format(f_name)
            absolute_file_path = f'{folder_path}/{file_name}' # 絕對路徑

            if z_int == ZZ:
                '''
                if z the same. Begine 
                '''
                chack_file_list[i].append(file_name)

                xdat_r, ydat_r          = read_file(absolute_file_path)

                try:
                    x_dat, y_dat            = skip_noise(xdat_r, ydat_r)
                    avg_x_byN, del_y_byN, N = gradient(x_dat, y_dat, 3)
                    mid_ofx, xpeaks         = Find_thepeak(x_dat, y_dat, N) # 與 gradient 綁在一起
                    D_1, D_2                = data_split_and_fit(x_dat, y_dat, xpeaks, N) # 與 Find_thepeak 綁在一起

                    k_parameters   = append_all_k(D_1, D_2, mid_ofx, find_midy(x_dat, y_dat))
                    spot_size_list = calculate_spot_size(k_parameters, speed)
                    for spot_i in spot_size_list:
                        spot_dependence_on_z[i].append( round(spot_i, 2) )
                except:
                    proplem_list.append(file_name)

    return spot_dependence_on_z, proplem_list


def FAS(file, get_row_data=None):
    '''
    file : put absolute path
    analysis and split
    '''
    if get_row_data == None:

        file_name       = absolute_to_relative(file)
        _, speed, z_int = judgment_format(file_name)


        xdat_row, ydat_row      = read_file(file)
        x_dat, y_dat            = skip_noise(xdat_row, ydat_row)
        x_dat, y_dat            = remove_top_bottom_noise(x_dat, y_dat)
        x_dat, y_dat            = remove_top_bottom_noise(x_dat, y_dat)
        avg_x_byN, del_y_byN, N = gradient(x_dat, y_dat, 3)
        mid_ofx, xpeaks         = Find_thepeak(x_dat, y_dat, N) # 與 gradient 綁在一起
        D_1, D_2                = data_split_and_fit(x_dat, y_dat, xpeaks, N) # 與 Find_thepeak 綁在一起
        return file_name, speed, z_int, xdat_row, ydat_row, x_dat, y_dat, avg_x_byN, del_y_byN, mid_ofx, xpeaks, D_1, D_2

    elif get_row_data == 'yes':
        file_name           = absolute_to_relative(file)
        xdat_row, ydat_row  = read_file(file)
        return file_name, xdat_row, ydat_row



def plt_all():

    # folder_path = '/Users/k.y.chen/Desktop/1122ok'     # 刪除
    # list_of_file = []   # 刪除
    # os.chdir(folder_path)   # 刪除
    # for file in glob.glob("*.txt"):   # 刪除
    #     if judgment_format(file) is not None:   # 刪除
    #         list_of_file.append(  judgment_format(file)[0]  )   # 刪除
    list_of_file, folder_path = r_all_file()
    
#     plt start:
    fig_N = len(list_of_file)
    row = math.ceil(fig_N/7) if math.ceil(fig_N/7) != 0 else 1
    fig, axes = plt.subplots(row, 7, figsize=(18, 2*row), dpi=100)
    fig.subplots_adjust(wspace = .45)  # hspace = .45, 
    axes = axes.ravel()
    c = ['#B22222', '#CD9B1D', '#FF7D40', '#FFC125', '#FF3030', '#FFC125', 
        '#B22222', '#CD9B1D', '#FF7D40', '#FFC125', '#FF3030', '#FFC125']

    for K, file_name in enumerate(list_of_file):
        absolute_file_path = f'{folder_path}/{file_name}' # 絕對路徑
        # file_name, xdat_row, ydat_row,= FAS(absolute_file_path, 'yes')
        
        file_name, speed, z_int, xdat_row, ydat_row, x_dat, y_dat, avg_x_byN, del_y_byN, mid_ofx, xpeaks, D_1, D_2 = FAS(absolute_file_path)
        k_parameters   = append_all_k(D_1, D_2, mid_ofx, find_midy(x_dat, y_dat))
        axes[K].axis('off')
        axes[K].set_title(f'{file_name}', fontsize=8)
        # axes[K].set_xlabel("time (s)", fontsize=8)
        # axes[K].set_ylabel("signal (a.u.)", fontsize=8)
        # axes[K].plot(x_dat, y_dat, 'b.',markersize = 2)
        axes[K].plot(xdat_row, ydat_row, 'b.',markersize = 2)
        
        if len(mid_ofx) == 1:
                p = fit_error(x_dat, y_dat, mid_ofx[0], find_midy(x_dat, y_dat))
                a, k, x0, y0 = p[0], p[1], p[2], p[3]
                axes[K].plot(x_dat, myerf(x_dat, a, k, x0, y0), color=c[0], linewidth=8, alpha=0.5)
                # plt.show()
        else:
            for i, val in enumerate(mid_ofx):
                p = fit_error(D_1[i], D_2[i], val, find_midy(x_dat, y_dat))
                a, k, x0, y0 = p[0], p[1], p[2], p[3]
                axes[K].plot(D_1[i], myerf(D_1[i], a, k, x0, y0), color=c[i] , linewidth=8, alpha=0.5)
    plt.show()

plt_all()




