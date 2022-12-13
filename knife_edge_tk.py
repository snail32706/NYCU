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

# file_path_bytk = None
xdat_row, ydat_row = None, None

def open_file(): 
    '''
    *** load data is the first step. ***
    use tkinter
    '''
    global speed, z_component
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
        # type(file_path_bytk) is str

    speed, z_component = found_name(file)

    rawdat = np.loadtxt(file_path_bytk, delimiter='\t')  
    xdat = rawdat[:,0]  
    ydat = rawdat[:,1]  
    
    return xdat, ydat, speed, z_component

def found_name(file):
    '''
    Use split to found file name.
    than get the speed & z component.
    '''
    file_name_all = str(file)

    for i in file_name_all.split('/'):
        if 'txt' in i:
            file_name, _ = i.split('txt')
            
    if file_name[0] != 's':
        # print('Name Error! \nFile name should be: \n\ts100_z10_ ...')
        return None, None
    else:
        for i, word in enumerate( file_name.split('_') ):
            if i == 0 and word[0] == 's':
                _, speed_local = word.split('s')
            elif i == 1 and word[0] == 'z':
                _, z_component_local = word.split('z')
        return int(speed_local), int(z_component_local)

def read_file(file_name):
    
    rawdat = np.loadtxt(file_name, delimiter='\t')  

    xdat = rawdat[:,0]
    ydat = rawdat[:,1]
    
    return xdat, ydat
# xdat_row, ydat_row = read_file(file_path_bytk)

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
    p0=[0.023, 80.0, mid_x, data_list[0]]
    popt, pcov = curve_fit(myerf, xdat, ydat, p0, method="lm")
    perr = np.sqrt(np.diag(pcov)) * 4.0
    σ = np.sqrt(np.diag(pcov))        
    a = pd.DataFrame(data={'params':popt,'σ':σ}, index = myerf.__code__.co_varnames[1:])      
    # display(a)   

    return popt

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
    return xx, yy, data_list  
# x_new, y_new, data_list = skip_noise(xdat_row ,ydat_row)

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

    return ave_x ,y_grad, y_grad_

def Find_thepeak(x_new, y_new, N):
    
    ave_x, y_grad, y_grad_= gradient(x_new, y_new, N)
    
    counts, dis = np.histogram(y_grad, bins= 5)
    # plt.stairs(counts, dis) # 可刪
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

    return x, peaks, ave_x, y_grad

def data_split_and_fit(N):
    
    mid_ofx, xpeaks, ave_x, y_grad = Find_thepeak(x_new, y_new, N)
    
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
    
    return D_1, D_2, mid_ofx, xpeaks, ave_x, y_grad
# D_1, D_2, mid_ofx, xpeaks, ave_x, y_grad = data_split_and_fit(3) # 改變平均值

spot_size_list = []
def calculate_spot_size(k_parameter):
    '''
    put all k
    calculate spot size and StD.
    '''
    for i, val in enumerate(k_parameter):
        k_um = val / speed 
        spot_size = 2*np.pi / ( k_um**2 ) # spot size = pi * r^2
        spot_size_list.append(spot_size)

'''
kife_edge.py end

'''
speed, z_component = None, None
x_new, y_new, data_list = None, None, None
D_1, D_2, mid_ofx, xpeaks, ave_x, y_grad = None, None, None, None, None, None

def B0f():
    '''
    load row data and show
    '''
    global xdat_row, ydat_row, x_new, y_new, data_list, D_1, D_2, mid_ofx, xpeaks, ave_x, y_grad, speed

    xdat_row, ydat_row, speed, z_component = open_file()
    x_new, y_new, data_list = skip_noise(xdat_row, ydat_row)    
    D_1, D_2, mid_ofx, xpeaks, ave_x, y_grad = data_split_and_fit(3) # 改變平均值
    ax.clear()
    ax.set_xlabel("time (s)")
    ax.set_ylabel("signal (a.u.)")
    ax.plot(xdat_row, ydat_row, 'b.',markersize = 4), ax.grid(True)
    line.draw() 

    # clear lower right corner
    r = tk.Label(root, bg='#C6C6C6') 
    r.place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2) 

    if speed is not None:
        l = tk.Label(root, fg='#FFDC00', font=("Arial", 18),
        text = f'Load Success!')
        l.place(relx=0.02, rely=0.85, relwidth=0.16, relheight=0.05)

    elif speed is None:
        var = tk.StringVar()
        errre_for_filename = tk.Label(root, textvariable=var, bg='#C6C6C6', fg='#F00000', font=("Arial", 18))
        var.set('Name Error! \nFile name should be: \n\ts100_z10_ ...')
        errre_for_filename .place(relx=0.26, rely=0.81)


def B1f():
    '''
    show processing data
    '''
    # x_new, y_new, data_list = skip_noise(xdat_row, ydat_row)    
    ax.clear()
    ax.set_xlabel("time (s)")
    ax.set_ylabel("signal (a.u.)")
    ax.plot(x_new, y_new, 'b.',markersize = 4), ax.grid(True)
    line.draw()   

def B2f():
    '''
    show gradient and peaks.
    '''

    ax.clear()
    ax.set_xlabel("time (s)")
    ax.set_ylabel("signal (a.u.)")
    ax.plot(ave_x, y_grad, 'b.-',markersize = 4), ax.grid(True)
    # ax.scatter(x, y, s=area, c=colors, alpha=0.5)
    for i in xpeaks:
        ax.scatter(ave_x[i], y_grad[i], s=8**2, c='r', alpha=0.5)
    line.draw()    


def B3f():

    k_parameter = []
    ax.clear()
    ax.plot(x_new, y_new, 'b.',markersize = 5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("signal (a.u.)")
    ax.set_title("fit: a * erf(k * (x - x0)) + y0")
    c = ['#B22222', '#CD9B1D', '#FF7D40', '#FFC125', '#FF3030', '#FFC125', 
        '#B22222', '#CD9B1D', '#FF7D40', '#FFC125', '#FF3030', '#FFC125'] # color code for fit diff curve
    if len(mid_ofx) == 1:
            p = fit_error(x_new, y_new, mid_ofx[0])
            a, k, x0, y0 = p[0], p[1], p[2], p[3]
            ax.plot(x_new, myerf(x_new, a, k, x0, y0), color=c[0], linewidth=8, alpha=0.5)
            k_parameter.append(p[1])
    else:    
        for i, val in enumerate(mid_ofx):
            p = fit_error(D_1[i], D_2[i], val)
            a, k, x0, y0 = p[0], p[1], p[2], p[3]
            ax.plot(D_1[i], myerf(D_1[i], a, k, x0, y0), color=c[i] , linewidth=8, alpha=0.5)
            k_parameter.append(p[1])
    line.draw()  

    if k_parameter is not None:
        '''
        caculate average spot size

        parameter k is slope which mean k = Δy / Δx.
        If Δx = time(s) * speed(um/s). 
        Slope k(1/um) = k(1/s) * (1/speed)
        '''
        calculate_spot_size(k_parameter)  # note: spot_size_list is global valuable.
        spot_size_avg = sum(spot_size_list) / len(spot_size_list)
        spot_size_StD = np.std(   np.array(spot_size_list)  )
        spot_size_StD_f = np.round(spot_size_StD, 2)

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

        spot_size_avg_f = round(spot_size_avg, 2)
        '''
        Avoid too long values.
        '''
        if float(spot_size_avg) > 100:
            spot_size_avg_f = round(spot_size_avg, 1)
        elif float(spot_size_avg) > 1000:
            spot_size_avg_f = round(spot_size_avg, 0)

        # print spot size:
        la4 = tk.Label(root, textvariable=var4, bg='#F0F0F0', fg='#BA1515', font=("Arial", 18), relief="ridge") 
        var4.set(f'{spot_size_avg_f} ± {spot_size_StD_f}')
        la4.place(relx=0.32, rely=0.87)

        la5 = tk.Label(root, textvariable=var5, bg='#C6C6C6', fg='#000000', font=("Arial", 18))
        var5.set("um/s")
        la5.place(relx=0.37, rely=0.81)

        la6 = tk.Label(root, textvariable=var6, bg='#C6C6C6', fg='#000000', font=("Arial", 18))
        var6.set("um^2")
        la6.place(relx=0.44, rely=0.87)

    # elif k_parameter == None:
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
    '''
    show table
    要優化全域變數

    '''
    # xdat_row, ydat_row, speed, z_component = open_file()
    # ax.plot(xdat_row, ydat_row, 'b.',markersize = 4), ax.grid(True)


    
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



