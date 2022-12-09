import tkinter as tk
import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# file_path_bytk = None
xdat_row, ydat_row = None, None
def open_file(): 
    '''
    *** load data is the first step. ***
    '''
    filetypes = ( ('text files', '*.txt'), ('All files', '*.*') )
    # global file_path_bytk
    # file = tk.filedialog.askopenfile(mode='r', filetypes=[('Python Files', '*.py')])
    file = tk.filedialog.askopenfile(
                            title='Open a file',
                            initialdir='/Users/k.y.chen/Desktop/', 
                            filetypes=filetypes)
    # file type is <class '_io.TextIOWrapper'> 
    if file:
        file_path_bytk = os.path.abspath(file.name)
        # file_path_bytk = str(file_path_bytk)
        # type(file_path_bytk) is str

    rawdat = np.loadtxt(file_path_bytk, delimiter='\t')  
    xdat = rawdat[:,0]
    ydat = rawdat[:,1]  
    return xdat, ydat

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
        ax1.set_xlabel("time (ms)")
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
#     σ = np.sqrt(np.diag(pcov))        
#     a = pd.DataFrame(data={'params':popt,'σ':σ}, index = myerf.__code__.co_varnames[1:])      
#     display(a)   

    return popt

def skip_noise():
    
    '''
    First:
    Find the noisce data. Useing histogram to find noise.
    
    Second:
    Using noise range (noise_bottom ~ noise_top) skip the noise data.
    '''
    # load file
    xdat, ydat = xdat_row, ydat_row 

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
# x_new, y_new, data_list = skip_noise()

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

'''
kife_edge.py end

'''
x_new, y_new, data_list = None, None, None
D_1, D_2, mid_ofx, xpeaks, ave_x, y_grad = None, None, None, None, None, None

def B0f():
    '''
    load row data and show
    '''
    global xdat_row, ydat_row, x_new, y_new, data_list, D_1, D_2, mid_ofx, xpeaks, ave_x, y_grad 
    xdat_row, ydat_row = open_file()
    x_new, y_new, data_list = skip_noise()
    D_1, D_2, mid_ofx, xpeaks, ave_x, y_grad = data_split_and_fit(3) # 改變平均值
    ax.clear()
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("signal (a.u.)")
    ax.plot(xdat_row, ydat_row, 'b.',markersize = 4), ax.grid(True)
    line.draw() 

def B1f():
    '''
    show processing data
    '''
    ax.clear()
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("signal (a.u.)")
    ax.plot(x_new, y_new, 'b.',markersize = 4), ax.grid(True)
    line.draw()   

def B2f():
    '''
    show gradient and peaks.
    '''
    ax.clear()
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("signal (a.u.)")
    ax.plot(ave_x, y_grad, 'b.-',markersize = 4), ax.grid(True)
    # ax.scatter(x, y, s=area, c=colors, alpha=0.5)
    for i in xpeaks:
        ax.scatter(ave_x[i], y_grad[i], s=8**2, c='r', alpha=0.5)
    line.draw()    

def B3f():

    ax.clear()
    ax.plot(x_new, y_new, 'b.',markersize = 5)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("signal (a.u.)")
    ax.set_title("fit: a * erf(k * (x - x0)) + c")
    c = ['#B22222', '#CD9B1D', '#FF7D40', '#FFC125', '#FF3030', '#FFC125', '#B22222', '#CD9B1D', '#FF7D40', '#FFC125', '#FF3030', '#FFC125'] # color code for fit diff curve
    if len(mid_ofx) == 1:
            p = fit_error(x_new, y_new, mid_ofx[0])
            a, k, x0, y0 = p[0], p[1], p[2], p[3]
            ax.plot(x_new, myerf(x_new, a, k, x0, y0), color=c[0] , linewidth=8, alpha=0.5)
    else:    
        for i, val in enumerate(mid_ofx):
            p = fit_error(D_1[i], D_2[i], val)

            a, k, x0, y0 = p[0], p[1], p[2], p[3]
            ax.plot(D_1[i], myerf(D_1[i], a, k, x0, y0), color=c[i] , linewidth=8, alpha=0.5)
    line.draw()  

def B4f():
    ax.clear()
    ax.axis('off')
    # ax.text(0.15, 0.55, 'Hello World!', fontsize=20, color='k')

    line.draw() 




#--- Raiz ---
root = tk.Tk()
root.geometry('900x600')
root.title("Error Func Fitting")
#------------

#-- Frames 框架 ---

'''
Frame:
    bd:     指標周圍邊框的大小
    height: The vertical dimension of the new frame.

place:
    Place 是幾何管理器是 Tkinter 中提供的三個通用幾何管理器中最簡單的一個。
    它允許您明確設置窗口的位置和大小，無論是絕對值還是相對於另一個窗口。

    relheight, relwidth: 高度和寬度作為 0.0 和 1.0 之間的浮點數
    relx, rely: 水平和垂直偏移量作為 0.0 和 1.0 之間的浮點數，作為父部件高度和寬度的一部分。
'''

left_frame = tk.Frame(root)
left_frame.place(relx=0.02, rely=0.02, relwidth=0.16, relheight=0.55)

right_frame = tk.Frame(root, bg='#C0C0C0') 
right_frame.place(relx=0.2, rely=0.02, relwidth=0.78, relheight=0.72)

ld_frame = tk.Frame(root, bg='#FFEBA4')
ld_frame.place(relx=0.02, rely=0.59, relwidth=0.16, relheight=0.39)

rd_frame = tk.Frame(root, bg='#C6C6C6')
rd_frame.place(relx=0.2, rely=0.78, relwidth=0.78, relheight=0.2)


#---------------

#--- Botones ---
RH = 0.19

B0 = tk.Button(left_frame,text='Load data' + '\n' 'and Show',command = B0f)
B0.place(relheight=RH, relwidth=1)

B1 = tk.Button(left_frame,text="Remove noise",command = B1f)
B1.place(rely=(0.1 + RH*0.54) ,relheight=RH, relwidth=1)

B2 = tk.Button(left_frame,text="Gradient",command = B2f)
B2.place(rely= 2*(0.1 + RH*0.54) ,relheight=RH, relwidth=1)

B3 = tk.Button(left_frame,text="Fit Error func",command = B3f)
B3.place(rely= 3*(0.1 + RH*0.54) ,relheight=RH, relwidth=1)

B4 = tk.Button(left_frame, text="Clear",command = B4f)
B4.place(rely= 4*(0.1 + RH*0.54) ,relheight=RH, relwidth=1)
#------------

#--- Agregar figura ---
figure = plt.Figure(figsize=(5,6), dpi=100)
ax = figure.add_subplot(111)
ax.grid(True),ax.set_xlabel('$x$'),ax.set_ylabel('$y(x)$')
line = FigureCanvasTkAgg(figure, right_frame)
line.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,expand=1)
#----------------------

root.mainloop()



