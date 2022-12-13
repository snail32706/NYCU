import tkinter as tk
from tkinter import ttk
import os
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
            list_of_file.append(  judgment_format(file)[0]  )
    '''
    list_of_file: [ 'file_name', 'file_name', 'file_name', ... ]
    '''
    return list_of_file, folder_path 

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

def absolute_to_relative(absolute_file_path):

    file_name = str(absolute_file_path)
    for i in file_name.split('/'):
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
    return xx, yy


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

def Find_thepeak(x_new, y_new, N):

    avg_x_byN, del_y_byN = gradient(x_new, y_new, N)[0], gradient(x_new, y_new, N)[1]
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

def data_split_and_fit(x_new, y_new, xpeaks, N):
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
    chack_file_list = []

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

                x_new, y_new            = skip_noise(xdat_r, ydat_r)
                avg_x_byN, del_y_byN, N = gradient(x_new, y_new, 3)
                mid_ofx, xpeaks         = Find_thepeak(x_new, y_new, N) # 與 gradient 綁在一起
                D_1, D_2                = data_split_and_fit(x_new, y_new, xpeaks, N) # 與 Find_thepeak 綁在一起

                k_parameters   = append_all_k(D_1, D_2, mid_ofx, find_midy(x_new, y_new))
                spot_size_list = calculate_spot_size(k_parameters, speed)

                for spot_i in spot_size_list:
                    spot_dependence_on_z[i].append( round(spot_i, 2) )
    return spot_dependence_on_z

def FAS(file, get_row_data=None):
    '''
    file : put absolute path
    analysis and split
    '''
    if get_row_data == None:
        file_name       = absolute_to_relative(file)
        _, speed, z_int = judgment_format(file_name)

        xdat_row, ydat_row      = read_file(file)
        x_new, y_new            = skip_noise(xdat_row, ydat_row)
        avg_x_byN, del_y_byN, N = gradient(x_new, y_new, 3)
        mid_ofx, xpeaks         = Find_thepeak(x_new, y_new, N) # 與 gradient 綁在一起
        D_1, D_2                = data_split_and_fit(x_new, y_new, xpeaks, N) # 與 Find_thepeak 綁在一起
        return file_name, speed, z_int, xdat_row, ydat_row, avg_x_byN, mid_ofx, D_1, D_2

    elif get_row_data == 'yes':
        file_name           = absolute_to_relative(file)
        xdat_row, ydat_row  = read_file(file)
        return file_name, xdat_row, ydat_row


# ------ ------------ ------ # 
# ------ tkinter func ------ # 

absolute_file_path = None # 唯一 global variable

def B0f():
    '''
    load row data and show
    '''
    global absolute_file_path

    absolute_file_path = open_file()
    file_name, xdat_row, ydat_row,= FAS(absolute_file_path, 'yes')
    ax.clear()
    ax.set_xlabel("time (s)")
    ax.set_ylabel("signal (a.u.)")
    ax.plot(xdat_row, ydat_row, 'b.',markersize = 4), ax.grid(True)
    line.draw() 

    # clear lower right corner
    r = tk.Label(root, bg='#C6C6C6') 
    r.place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2) 

    l = tk.Label(root, fg='#FFDC00', font=("Arial", 18),
    text = f'Load Success!')
    l.place(relx=0.02, rely=0.85, relwidth=0.16, relheight=0.05)

def B1f():
    '''
    show processing data
    '''  

    file_name, speed, z_int, xdat_row, ydat_row, avg_x_byN, mid_ofx, D_1, D_2 = FAS(absolute_file_path)

    ax.clear()
    ax.set_xlabel("time (s)")
    ax.set_ylabel("signal (a.u.)")
    ax.plot(x_new, y_new, 'b.',markersize = 4), ax.grid(True)
    line.draw()   

def B2f():
    '''
    show gradient and peaks.
    '''
    file_name, speed, z_int, xdat_row, ydat_row, avg_x_byN, mid_ofx, D_1, D_2 = FAS(absolute_file_path)

    ax.clear()
    ax.set_xlabel("time (s)")
    ax.set_ylabel("signal (a.u.)")
    ax.plot(avg_x_byN, abs(del_y_byN), 'b.-',markersize = 4), ax.grid(True)
    # ax.scatter(x, y, s=area, c=colors, alpha=0.5)
    for i in mid_ofx:
        ax.scatter(avg_x_byN[i], abs(del_y_byN)[i], s=8**2, c='r', alpha=0.5)
    line.draw()    

def B3f():
    
    '''
    Fit Error func
    '''
    file_name, speed, z_int, xdat_row, ydat_row, avg_x_byN, mid_ofx, D_1, D_2 = FAS(absolute_file_path)
    k_parameters   = append_all_k(D_1, D_2, mid_ofx, find_midy(x_new, y_new))
    spot_size_list = calculate_spot_size(k_parameters, speed)

    ax.clear()
    ax.plot(x_new, y_new, 'b.',markersize = 5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("signal (a.u.)")
    ax.set_title("fit: a * erf(k * (x - x0)) + y0")
    c = ['#B22222', '#CD9B1D', '#FF7D40', '#FFC125', '#FF3030', '#FFC125', 
        '#B22222', '#CD9B1D', '#FF7D40', '#FFC125', '#FF3030', '#FFC125'] # color code for fit diff curve
    if len(mid_ofx) == 1:
            p = fit_error(x_new, y_new, mid_ofx[0], find_midy(x_new, y_new))
            a, k, x0, y0 = p[0], p[1], p[2], p[3]
            ax.plot(x_new, myerf(x_new, a, k, x0, y0), color=c[0], linewidth=8, alpha=0.5)
    else:    
        for i, val in enumerate(mid_ofx):
            p = fit_error(D_1[i], D_2[i], val, find_midy(x_new, y_new))
            a, k, x0, y0 = p[0], p[1], p[2], p[3]
            ax.plot(D_1[i], myerf(D_1[i], a, k, x0, y0), color=c[i] , linewidth=8, alpha=0.5)
    line.draw()  

    if k_parameters is not None:

        all_spots = np.array(spot_size_list)
        spot_size_Avg = round( sum(all_spots) / len(all_spots), 2)
        spot_size_StD = round( np.std(all_spots), 2)
        N = len(all_spots)


        var1, var2, var3, var4, var5, var6 = tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar()
        la1 = tk.Label(root, textvariable=var1, bg='#C6C6C6', fg='#000000', font=("Arial", 18))
        var1.set("speed    : ")
        la1.place(relx=0.22, rely=0.81)
        la2 = tk.Label(root, textvariable=var2, bg='#C6C6C6', fg='#000000', font=("Arial", 18))
        var2.set("spot size: ")
        la2.place(relx=0.22, rely=0.87)
        la3 = tk.Label(root, textvariable=var3, bg='#F0F0F0', fg='#850000', font=("Arial", 18), relief="ridge")
        var3.set(speed)
        la3.place(relx=0.32, rely=0.81)


        '''
        Avoid too long values.
        '''
        if float(spot_size_Avg) > 100:
            spot_size_Avg = round(spot_size_Avg, 1)
        elif float(spot_size_Avg) > 1000:
            spot_size_Avg = round(spot_size_Avg, 0)

        # print spot size:
        la4 = tk.Label(root, textvariable=var4, bg='#F0F0F0', fg='#BA1515', font=("Arial", 18), relief="ridge") 
        var4.set(f'{spot_size_Avg} ± {spot_size_StD}')
        la4.place(relx=0.32, rely=0.87)

        la5 = tk.Label(root, textvariable=var5, bg='#C6C6C6', fg='#000000', font=("Arial", 18))
        var5.set("um/s")
        la5.place(relx=0.37, rely=0.81)

        la6 = tk.Label(root, textvariable=var6, bg='#C6C6C6', fg='#000000', font=("Arial", 18))
        var6.set("um^2")
        la6.place(relx=0.44, rely=0.87)

    # elif k_parameters == None:
        r.place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2)

def B4f():
    # cls

    ax.clear()
    ax.axis('off')
    line.draw() 

    r = tk.Label(root, bg='#C6C6C6') 
    r.place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2) 

def B5f():
    pass



    
#--- Raiz ---
root = tk.Tk()
root.geometry('900x600')
root.title("Error Func Fitting")
#------------

#-- Frames 框架 ---

'''
Frame:
    bd    : 指標周圍邊框的大小
    height: The vertical dimension of the new frame.

place:
    Place 是幾何管理器是 Tkinter 中提供的三個通用幾何管理器中最簡單的一個。
    它允許您明確設置窗口的位置和大小，無論是絕對值還是相對於另一個窗口。

    relheight, relwidth: 高度和寬度作為 0.0 和 1.0 之間的浮點數
    relx, rely: 水平和垂直偏移量作為 0.0 和 1.0 之間的浮點數，作為父部件高度和寬度的一部分。
'''

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
RH = 0.15

B0 = tk.Button(left_frame,text='Load data' + '\n' 'and Show',command = B0f)
B0.place(relheight=RH, relwidth=1)

B1 = tk.Button(left_frame,text="Remove noise",command = B1f)
B1.place(rely=(0.1 + RH*0.44) ,relheight=RH, relwidth=1)

B2 = tk.Button(left_frame,text="Gradient",command = B2f)
B2.place(rely= 2*(0.1 + RH*0.44) ,relheight=RH, relwidth=1)

B3 = tk.Button(left_frame,text="Fit Error func",command = B3f)
B3.place(rely= 3*(0.1 + RH*0.44) ,relheight=RH, relwidth=1)

B4 = tk.Button(left_frame, text="Clear",command = B4f)
B4.place(rely= 4*(0.1 + RH*0.44) ,relheight=RH, relwidth=1)

B5 = tk.Button(left_frame, text="Nothing",command = B5f)
B5.place(rely= 5*(0.1 + RH*0.44) ,relheight=RH, relwidth=1)
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



