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
    root = tk.Tk()
    root.withdraw()
    folder_path = tk.filedialog.askdirectory()
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

file_name = '/Users/k.y.chen/Desktop/s100_z10_-00-00-0101190 i assume.txt'

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
    # plt_xy(x_dat, y_dat, '.')
    return x_dat, y_dat











read_file(file_name)
skip_noise(xdat, ydat)






