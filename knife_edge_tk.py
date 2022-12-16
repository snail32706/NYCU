import tkinter as tk
from tkinter import ttk
import glob, os
import platform
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from IPython.display import display, Math, Latex
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import math
from PIL import Image, ImageTk


# ------ --------- ------ #
# ------ Load File ------ #

def rename():
    
#     folder_path = '/Users/k.y.chen/Desktop/huan file/20221118'     # 刪除

    folder_path = load_path()

    list_of_file = []
    os.chdir(folder_path)
    i = 17
    for file in glob.glob("*.txt"):
        old_name = file
        new_name = f'{folder_path}/s100_z{i}_{old_name}'
        os.rename(old_name, new_name)
        i += 1

def open_file(): 
    '''
    使用相對路徑
    *** load data is the first step. ***
    use tkinter
    '''

    filetypes = ( ('text files', '*.txt'), ('All files', '*.*') )
    # global file_path_bytk
    file = tk.filedialog.askopenfile(
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
    # root = tk.Tk()
    # root.withdraw()
    folder_path = tk.filedialog.askdirectory()
    return folder_path

def judgment_format(file):
    '''
    split flie name by '_'
    [ file_name, speed, z ]  >> type: str, int, int
    '''
    file_name = str(file)
    try:
        if file_name[0] == 's':
            SS, ZZ, _ = file_name.split('_')
            _ , speed = SS.split('s')
            _ , z_component = ZZ.split('z')
            return file_name, int(speed), int(z_component)
    except:
        pass

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

# def skip_noise(xdat, ydat):

#     # First step:
#     counts, dis = np.histogram(ydat, bins= 15)
#      # plt.stairs(counts, dis) # 可刪
#     region = []
#     for i, val in enumerate(counts):
#         if val > 50:
#             region.append(dis[i])
#             region.append(dis[i+1])
#     N = int(len(region)/2)
#     s_x, s_y = [], []
#     for i in range(N):
#         s_x.append([])
#         s_y.append([]) # f'region{i+1}:'
    
#     for i, val in enumerate(ydat):
#         for j in range(N):
#             if j == 0:
#                 if val < region[2*j+1]:
#                     s_x[j].append(xdat[i])
#                     s_y[j].append(ydat[i]) 
#             elif j == N-1:
#                 if val > region[2*j]:
#                     s_x[j].append(xdat[i])
#                     s_y[j].append(ydat[i]) 
#             else:
#                 if val > region[2*j] and val < region[2*j+1]:
#                     s_x[j].append(xdat[i])
#                     s_y[j].append(ydat[i]) 
#     noise = []
#     for j in range(N):
#         delta = 0
#         for i, val in enumerate(s_x[j]):
#             if i == 0:
#                 delta = abs(s_x[j][i] - xdat[0]) if abs(s_x[j][i] - min(xdat)) > abs(s_x[j][len(s_x[j])-1] - xdat[len(xdat)-1]) else abs(s_x[j][len(s_x[j])-1] - xdat[len(xdat)-1])
#             else:    
#                 delta = abs(s_x[j][i] - s_x[j][i-1]) if abs(s_x[j][i] - s_x[j][i-1]) > delta else delta
#         if delta < (max(xdat) - min(xdat)) / 100:
#             if len(noise) == 0:
#                 noise = s_x[j]
#             else:
#                 noise += s_x[j]
                
#     x_new, y_new = [], []       
#     for i, val in enumerate(xdat):  
#         if val not in noise:
#             x_new.append(val)
#             y_new.append(ydat[i])
#     return x_new, y_new

def skip_noise(xdat, ydat):
    
    # First step:
    N_split = int(10)
    counts, dis = np.histogram(ydat, bins= N_split)   
#     plt.stairs(counts, dis) # 可刪
    _y = counts.tolist()
    max_index = _y.index(max(counts)) 
    
    noise_top, noise_bottom = dis[max_index+1], dis[max_index]
    
    # Second step:
    xx, yy = [], []
    if max_index == N_split-1:
        for i, val in enumerate(ydat):
            if val > noise_bottom:
                pass
            else:
                xx.append(xdat[i])
                yy.append(ydat[i]) 
    elif max_index == 0:
        for i, val in enumerate(ydat):
            if val < noise_top:
                pass
            else:
                xx.append(xdat[i])
                yy.append(ydat[i]) 
    return xx, yy

def remove_top_bottom_noise(x_dat, y_dat):
    '''
    x_dat, y_dat is list. Result by "append"
    '''
    CC = 0
    while True:
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
        elif CC > 10:
            break
    # plt_xy(x_dat, y_dat, '.')
    # print(CC)
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
                    x_dat, y_dat            = remove_top_bottom_noise(x_dat, y_dat)
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

def plt_all(list_of_file, folder_path):
    
    # plt start:
    fig_N = len(list_of_file)
    row = math.ceil(fig_N/7) if math.ceil(fig_N/7) != 0 else 1
    fig, axes = plt.subplots(row, 7, figsize=(18, 2*row), dpi=100)
    fig.subplots_adjust(hspace = .15, wspace = .6)  # 
    axes = axes.ravel()
    c = ['#B22222', '#CD9B1D', '#FF7D40', '#FFC125', '#FF3030', '#FFC125', 
        '#B22222', '#CD9B1D', '#FF7D40', '#FFC125', '#FF3030', '#FFC125']

    for K, file_name in enumerate(list_of_file):
        absolute_file_path = f'{folder_path}/{file_name}' # 絕對路徑

        Name_i, Name_f = file_name[ :int(len(file_name)/2)], file_name[int(len(file_name)/2): ]
        axes[K].set_title(f'{Name_i}\n{Name_f}', fontsize=8)
        try: 
            file_name, speed, z_int, xdat_row, ydat_row, x_dat, y_dat, avg_x_byN, del_y_byN, mid_ofx, xpeaks, D_1, D_2 = FAS(absolute_file_path, plt_all='y')
            k_parameters   = append_all_k(D_1, D_2, mid_ofx, find_midy(x_dat, y_dat))
            axes[K].axis('off')
            # axes[K].set_title(f'{file_name}', fontsize=8)
            axes[K].plot(xdat_row, ydat_row, 'k.',markersize = 0.5)
            axes[K].plot(x_dat, y_dat, 'k.',markersize = 2)
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
        except:
            file_name, xdat_row, ydat_row,= FAS(absolute_file_path, get_row_data='yes', plt_all='y')
            axes[K].plot(xdat_row, ydat_row, 'b.',markersize = 2)
            axes[K].axis('off')

    plt.show()


def FAS(file, get_row_data=None, plt_all=None):
    '''
    file : put absolute path
    analysis and split
    '''
    file_name       = absolute_to_relative(file)
    xdat_row, ydat_row      = read_file(file)
    
    if plt_all is None:
        global x_new_G, y_new_G
    else:
        x_new_G, y_new_G = None, None

    if get_row_data == None:
        _, speed, z_int = judgment_format(file_name)        
        
        if x_new_G is None:
            x_dat, y_dat            = skip_noise(xdat_row, ydat_row)
            x_dat, y_dat            = remove_top_bottom_noise(x_dat, y_dat)
            x_new_G, y_new_G        = remove_top_bottom_noise(x_dat, y_dat)
            avg_x_byN, del_y_byN, N = gradient(x_new_G, y_new_G, 3)
            mid_ofx, xpeaks         = Find_thepeak(x_new_G, y_new_G, N) # 與 gradient 綁在一起
            D_1, D_2                = data_split_and_fit(x_new_G, y_new_G, xpeaks, N) # 與 Find_thepeak 綁在一起
            return file_name, speed, z_int, xdat_row, ydat_row, x_new_G, y_new_G, avg_x_byN, del_y_byN, mid_ofx, xpeaks, D_1, D_2
        else:
            avg_x_byN, del_y_byN, N = gradient(x_new_G, y_new_G, 3)
            mid_ofx, xpeaks         = Find_thepeak(x_new_G, y_new_G, N) # 與 gradient 綁在一起
            D_1, D_2                = data_split_and_fit(x_new_G, y_new_G, xpeaks, N) # 與 Find_thepeak 綁在一起
            return file_name, speed, z_int, xdat_row, ydat_row, x_new_G, y_new_G, avg_x_byN, del_y_byN, mid_ofx, xpeaks, D_1, D_2
        
    elif get_row_data == 'yes':
        return file_name, xdat_row, ydat_row


# ------ ------------ ------ # 
# ------ tkinter func ------ # 

x_new_G, y_new_G = None, None
absolute_file_path = None # 唯一 global variable

def B0f():
    '''
    load row data and show
    '''
    global absolute_file_path, x_new_G, y_new_G

    try:
        absolute_file_path = open_file()
        file_name, xdat_row, ydat_row,= FAS(absolute_file_path, get_row_data='yes')
        # x_new_G, y_new_G = None, None
        ax.clear()
        ax.set_xlabel("time (s)")
        ax.set_ylabel("signal (a.u.)")
        ax.plot(xdat_row, ydat_row, 'b.',markersize = 4), ax.grid(True)
        line.draw() 
    except:
        absolute_file_path = None
        x_new_G, y_new_G = None, None
        ax.clear()
        line.draw() 

    # clear lower right corner
    r = tk.Label(root, bg='#C6C6C6') 
    r.place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2) 

    if absolute_file_path is not None:
        if platform.system() == 'Windows':
            l = tk.Label(root, fg='#27AE60', font=("Arial", 15),
                text = f'Load Success!').place(relx=0.02, rely=0.85, relwidth=0.16, relheight=0.05)
        else:
            l = tk.Label(root, fg='#FFDC00', font=("Arial", 15),
                text = f'Load Success!').place(relx=0.02, rely=0.85, relwidth=0.16, relheight=0.05)
        r = tk.Label(root, bg='#C6C6C6', fg='#000000', font=("Arial", 15),
            text = f'file name:\n{file_name}').place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2) 
    else:
        tk.Label(root, text = '').place(relx=0.02, rely=0.85, relwidth=0.16, relheight=0.05)
        tk.Label(root, bg='#C6C6C6').place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2) 

def B1f():
    '''
    show processing data
    '''  
    if absolute_file_path is None:
        tk.Label(root, text = 'load another file').place(relx=0.02, rely=0.85, relwidth=0.16, relheight=0.05)
        tk.Label(root, bg='#C6C6C6').place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2) 
        open_popup1()
    try:
        file_name, speed, z_int, xdat_row, ydat_row, x_dat, y_dat, avg_x_byN, del_y_byN, mid_ofx, xpeaks, D_1, D_2 = FAS(absolute_file_path)
        ax.clear()
        ax.set_xlabel("time (s)")
        ax.set_ylabel("signal (a.u.)")
        ax.plot(x_dat, y_dat, 'b.',markersize = 4), ax.grid(True)
        line.draw()   
    except:
        file_name = absolute_to_relative(absolute_file_path)
        if judgment_format(file_name) is None:
            open_popup3(file_name)
        else:
            tk.Label(root, text = 'load another file').place(relx=0.02, rely=0.85, relwidth=0.16, relheight=0.05)
            tk.Label(root, bg='#C6C6C6').place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2) 
            open_popup2(file_name)  

def B2f():
    '''
    show gradient and peaks.
    '''
    if absolute_file_path is None:
        tk.Label(root, text = 'load another file').place(relx=0.02, rely=0.85, relwidth=0.16, relheight=0.05)
        tk.Label(root, bg='#C6C6C6').place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2) 
        open_popup1()
    try:
        file_name, speed, z_int, xdat_row, ydat_row, x_dat, y_dat, avg_x_byN, del_y_byN, mid_ofx, xpeaks, D_1, D_2 = FAS(absolute_file_path)
        ax.clear()
        ax.set_xlabel("time (s)")
        ax.set_ylabel("signal (a.u.)")
        ax.plot(avg_x_byN, abs(del_y_byN), 'b.-',markersize = 4), ax.grid(True)
        # ax.scatter(x, y, s=area, c=colors, alpha=0.5)
        for i in xpeaks:
            ax.scatter(avg_x_byN[i], abs(del_y_byN)[i], s=8**2, c='r', alpha=0.5)
        line.draw()    
    except:
        file_name = absolute_to_relative(absolute_file_path)
        if judgment_format(file_name) is None:
            open_popup3(file_name)
        else:
            tk.Label(root, text = 'load another file').place(relx=0.02, rely=0.85, relwidth=0.16, relheight=0.05)
            tk.Label(root, bg='#C6C6C6').place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2) 
            open_popup2(file_name)  

def B3f():
    
    '''
    Fit Error func
    '''
    tk.Label(root, bg='#C6C6C6').place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2)  # cls 
    if absolute_file_path is None:
        tk.Label(root, text = 'load another file').place(relx=0.02, rely=0.85, relwidth=0.16, relheight=0.05)
        tk.Label(root, bg='#C6C6C6').place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2) 
        open_popup1()
    try:
        file_name, speed, z_int, xdat_row, ydat_row, x_dat, y_dat, avg_x_byN, del_y_byN, mid_ofx, xpeaks, D_1, D_2 = FAS(absolute_file_path)
        k_parameters   = append_all_k(D_1, D_2, mid_ofx, find_midy(x_dat, y_dat))
        spot_size_list = calculate_spot_size(k_parameters, speed)

        ax.clear()
        ax.plot(x_dat, y_dat, 'b.',markersize = 5)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("signal (a.u.)")
        ax.set_title("fit: a * erf(k * (x - x0)) + y0")
        c = ['#B22222', '#CD9B1D', '#FF7D40', '#FFC125', '#FF3030', '#FFC125', 
            '#B22222', '#CD9B1D', '#FF7D40', '#FFC125', '#FF3030', '#FFC125'] # color code for fit diff curve
        if len(mid_ofx) == 1:
                p = fit_error(x_dat, y_dat, mid_ofx[0], find_midy(x_dat, y_dat))
                a, k, x0, y0 = p[0], p[1], p[2], p[3]
                # ax.plot(x_dat, myerf(x_dat, a, k, x0, y0), color=c[0], linewidth=8, alpha=0.5)
                ax.plot(x_dat, myerf(x_dat, a, k, x0, y0), color='r', linewidth=10, alpha=0.5)
        else:    
            for i, val in enumerate(mid_ofx):
                p = fit_error(D_1[i], D_2[i], val, find_midy(x_dat, y_dat))
                a, k, x0, y0 = p[0], p[1], p[2], p[3]
                ax.plot(D_1[i], myerf(D_1[i], a, k, x0, y0), color=c[i] , linewidth=8, alpha=0.5)
        line.draw()  
    except:
        file_name = absolute_to_relative(absolute_file_path)
        if judgment_format(file_name) is None:
            open_popup3(file_name)
        else:
            tk.Label(root, text = 'load another file').place(relx=0.02, rely=0.85, relwidth=0.16, relheight=0.05)
            tk.Label(root, bg='#C6C6C6').place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2) 
            open_popup2(file_name)    

    if k_parameters is not None:

        all_spots = np.array(spot_size_list)
        spot_size_Avg = round( sum(all_spots) / len(all_spots), 2)
        spot_size_StD = round( np.std(all_spots), 2)
        N = len(all_spots)

        # r = tk.Label(root, bg='#C6C6C6', fg='#000000', font=("Arial", 15),
        #     text = f'file name: {file_name}').place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2)
        

        var0 ,var1, var2, var3, var4, var5, var6 = tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar()
        la0 = tk.Label(root, textvariable=var0, bg='#C6C6C6', fg='#000000', font=("Arial", 14)).place(relx=0.24, rely=0.79)
        var0.set(f'file name: {file_name}')
        la1 = tk.Label(root, textvariable=var1, bg='#C6C6C6', fg='#000000', font=("Arial", 18)).place(relx=0.22, rely=0.85)
        var1.set("speed    : ")
        la2 = tk.Label(root, textvariable=var2, bg='#C6C6C6', fg='#000000', font=("Arial", 18)).place(relx=0.22, rely=0.9)
        var2.set("spot size: ")

        if platform.system() == 'Darwin':

            la3 = tk.Label(root, textvariable=var3, bg='#F0F0F0', fg='#850000', font=("Arial", 18), relief="ridge").place(relx=0.32, rely=0.85)
            var3.set(speed)

            '''
            Avoid too long values.
            '''
            if float(spot_size_Avg) > 100:
                spot_size_Avg = round(spot_size_Avg, 1)
            elif float(spot_size_Avg) > 1000:
                spot_size_Avg = round(spot_size_Avg, 0)
            if float(spot_size_Avg) > 100:
                spot_size_Avg = round(spot_size_Avg, 1)
            elif float(spot_size_Avg) > 1000:
                spot_size_Avg = round(spot_size_Avg, 0)

            # print spot size:
            la4 = tk.Label(root, textvariable=var4, bg='#F0F0F0', fg='#BA1515', font=("Arial", 18), relief="ridge").place(relx=0.32, rely=0.9)
            var4.set(f'{spot_size_Avg} ± {spot_size_StD}')

            la5 = tk.Label(root, textvariable=var5, bg='#C6C6C6', fg='#000000', font=("Arial", 18)).place(relx=0.37, rely=0.85)
            var5.set("um/s")

            la6 = tk.Label(root, textvariable=var6, bg='#C6C6C6', fg='#000000', font=("Arial", 18)).place(relx=0.392, rely=0.9)
            var6.set("um^2")

        elif platform.system() == 'Windows':

            la3 = tk.Label(root, textvariable=var3, bg='#F0F0F0', fg='#850000', font=("Arial", 18), relief="ridge").place(relx=0.34, rely=0.85)
            var3.set(speed)
            '''
            Avoid too long values.
            '''
            if float(spot_size_Avg) > 100:
                spot_size_Avg = round(spot_size_Avg, 1)
            elif float(spot_size_Avg) > 1000:
                spot_size_Avg = round(spot_size_Avg, 0)

            # print spot size:
            la4 = tk.Label(root, textvariable=var4, bg='#F0F0F0', fg='#BA1515', font=("Arial", 18), relief="ridge").place(relx=0.34, rely=0.9)
            var4.set(f'{spot_size_Avg} ± {spot_size_StD}')

            la5 = tk.Label(root, textvariable=var5, bg='#C6C6C6', fg='#000000', font=("Arial", 18)).place(relx=0.4, rely=0.85)
            var5.set("um/s")

            la6 = tk.Label(root, textvariable=var6, bg='#C6C6C6', fg='#000000', font=("Arial", 18)).place(relx=0.49, rely=0.9)
            var6.set("um^2")

    # elif k_parameters == None:
        # r.place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2)

# def B4f():
#     # cls

#     ax.clear()
#     ax.axis('off')
#     line.draw() 

#     tk.Label(root, text = '').place(relx=0.02, rely=0.85, relwidth=0.16, relheight=0.05)
#     tk.Label(root, bg='#C6C6C6').place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2) 

def B4f():
    list_of_file, folder_path = r_all_file()
    if len(list_of_file) == 0:
        top = tk.Toplevel(root)
        top.geometry("500x250")
        top.title("File Name Error!")
        word = f'File Name Error!\nfile name should be: s100_z10_...'
        tk.Label(top, fg='#FF0000', text= word , font=('Mistral 18 bold')).place(x=250,y=125, anchor="center")
    else:
        plt_all(list_of_file, folder_path)


def B5f():
    # cls first
    tk.Label(root, text = '').place(relx=0.02, rely=0.85, relwidth=0.16, relheight=0.05)
    tk.Label(root, bg='#C6C6C6').place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2) 

    spot_dependence_on_z, proplem_list = main()
    row_N = len(spot_dependence_on_z)

    top = tk.Toplevel(root)
    top.geometry("400x550")
    top.title("Dependence of the spot size on the lens focal length")

    # Create an object of Style widget
    style = ttk.Style()
    style.theme_use('clam')

    # Add a Treeview widget
    style.configure('Treeview.Heading', foreground='black', background='white', font=('Arial',18))
    style.configure('Treeview', rowheight=20)
    style.configure('Treeview', foreground='black', background='white', font=('Arial',16),)

    # Add a Treeview widget
    tree = ttk.Treeview(top, columns=("1", "2", "3", "4"), 
                        show='headings', height=50)     # height: 多少列
    tree.column( "# 1", minwidth=0, width=80, anchor='center')
    tree.heading("# 1", text  ="z (um)")
    tree.column( "# 2", minwidth=0, width=130, anchor='center')
    tree.heading("# 2", text  ="spot size")
    tree.column( "# 3", minwidth=0, width=90, anchor='center')
    tree.heading("# 3", text  ="StD.")
    tree.column( "# 4", minwidth=0, width=50, anchor='center')
    tree.heading("# 4", text  ="N")

    for i, list_i in enumerate(spot_dependence_on_z):
        Z_vlal = list_i.pop(0)
        all_spots = np.array(list_i)
        spot_size_Avg = round( sum(all_spots) / len(all_spots), 2)
        spot_size_StD = round( np.std(all_spots), 2)
        N = len(all_spots)

        tree.insert('', 'end', text=str(i+1), 
                    values=(str(Z_vlal), str(spot_size_Avg), str(spot_size_StD), str(N))
                    )
    tree.pack()


# --- pop window --- #
def open_popup1():
   top = tk.Toplevel(root)
   top.geometry("300x250")
   top.title("Error!")
   tk.Label(top, fg='#FF0000', text= 'Plz load file first', 
            font=('Mistral 20 bold')).place(x=150,y=125, anchor="center")

def open_popup2(file_name):
   top = tk.Toplevel(root)
   top.geometry("500x250")
   top.title("Analyze Error!")
   word = f'{file_name}\nAnalyze Error!'
   tk.Label(top, fg='#FF0000', text= word , font=('Mistral 18 bold')).place(x=250,y=125, anchor="center")

def open_popup3(file_name):
   top = tk.Toplevel(root)
   top.geometry("500x250")
   top.title("Name Error!")
   word = f'{file_name}\nName Error!'
   tk.Label(top, fg='#FF0000', text= word , font=('Mistral 18 bold')).place(x=250,y=125, anchor="center")

#--- Raiz ---
root = tk.Tk()
root.geometry('900x600')
root.title("Error Func Fitting")
#------------

#-- Frames 框架 ---
# Im_load_file = tk.PhotoImage(file='/Users/k.y.chen/Documents/Coding_all/NYCU/load_file.jpeg')
root.photo = ImageTk.PhotoImage(Image.open('load_file.jpg'))

# '''
# Frame:
#     bd    : 指標周圍邊框的大小
#     height: The vertical dimension of the new frame.
# place:
#     Place 是幾何管理器是 Tkinter 中提供的三個通用幾何管理器中最簡單的一個。
#     它允許您明確設置窗口的位置和大小，無論是絕對值還是相對於另一個窗口。
#     relheight, relwidth: 高度和寬度作為 0.0 和 1.0 之間的浮點數
#     relx, rely: 水平和垂直偏移量作為 0.0 和 1.0 之間的浮點數，作為父部件高度和寬度的一部分。
# '''

left_frame = tk.Frame(root)
left_frame.place(relx=0.02, rely=0.02, relwidth=0.16, relheight=0.74)

right_frame = tk.Frame(root, bg='#C0C0C0') 
right_frame.place(relx=0.2, rely=0.02, relwidth=0.78, relheight=0.72)

# ld_frame = tk.Frame(root, bg='#FFEBA4')
# ld_frame.place(relx=0.02, rely=0.59, relwidth=0.16, relheight=0.39)

l = tk.Label(root, fg='#FF0000', font=("Arial", 15),
            text = f'Plz load data first!')
l.place(relx=0.02, rely=0.85, relwidth=0.16, relheight=0.05)

r = tk.Label(root, bg='#C6C6C6', fg='#000000', font=("Arial", 15))
r.place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2)

# rd_frame = tk.Frame(root, bg='#C6C6C6')
# rd_frame.place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2)
# tk.Label

#---------------

#--- Botones ---
RH = 0.13

B0 = tk.Button(left_frame,text='Load data' + '\n' 'and Show',command = B0f, relief="raised", image=root.photo)
B0.place(relheight=RH, relwidth=1)

B1 = tk.Button(left_frame,text="Remove noise",command = B1f)
B1.place(rely=(0.1 + RH*0.392) ,relheight=RH, relwidth=1)

B2 = tk.Button(left_frame,text="Gradient",command = B2f)
B2.place(rely= 2*(0.1 + RH*0.392) ,relheight=RH, relwidth=1)

B3 = tk.Button(left_frame,text="Fit Error func",command = B3f)
B3.place(rely= 3*(0.1 + RH*0.392) ,relheight=RH, relwidth=1)

B4 = tk.Button(left_frame, text="Plt all",command = B4f, relief="sunken")
B4.place(rely= 4*(0.1 + RH*0.392) ,relheight=RH, relwidth=1)

B5 = tk.Button(left_frame, text="Dependence\nof A on z",command = B5f, relief="groove")
B5.place(rely= 5*(0.1 + RH*0.392) ,relheight=RH, relwidth=1)
#------------

#--- Agregar figura ---
figure = plt.Figure(figsize=(5,6), dpi=100)
ax = figure.add_subplot(111)
ax.grid(True),ax.set_xlabel('$x$'),ax.set_ylabel('$y(x)$')
line = FigureCanvasTkAgg(figure, right_frame)
line.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,expand=1)
#----------------------
# B0f()
root.mainloop()