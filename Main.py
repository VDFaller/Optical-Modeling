"""
UI Program to create an optical modeler

THINGS TO DO:
Change join to merge and use wv not as an index
add wavelength to UI
make EMA dynamic
make fit dynamic
weight the fit to extrema
    use a flat weighting vector with gaussians at extrema
    parameterize the gaussians
sav golay the data?
change the brug tranform to return a df

"""
import sys
import os
import MainUI
import tmm_core as tmm
import tkinter as tk
from tkinter.filedialog import askopenfilename
from PyQt5 import QtWidgets
import pandas as pd
import numpy as np
from math import factorial
import scipy as sp
import matplotlib
from numpy.core.numeric import inf
from scipy.interpolate import interp1d
matplotlib.use("Qt5Agg")  # required currently because matplotlib uses PyQt4
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import lmfit


class MPLibWidget(QtWidgets.QWidget):
    """
    base MatPlotLib widget
    """
    def __init__(self, parent=None):
        super(MPLibWidget, self).__init__(parent)
        
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setParent(self)
        
        self.mpl_toolbar = NavigationToolbar2QT(self.canvas, self)
        
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        self.axes = self.figure.add_subplot(111)
        # self.axes.hold(False)

        self.compute_initial_figure()
        
        self.layoutVertical = QtWidgets.QVBoxLayout(self)
        self.layoutVertical.addWidget(self.canvas)
        self.layoutVertical.addWidget(self.mpl_toolbar)
        
    def on_key_press(self, event):
        """not working"""
        print('you pressed', event.key)
        # implement the default mpl key press events described at
        # http://matplotlib.org/users/navigation_toolbar.html#navigation-keyboard-shortcuts
        key_press_handler(event, self.canvas, self.mpl_toolbar)    
        
    def compute_initial_figure(self):
        pass


class MW(QtWidgets.QMainWindow, MainUI.Ui_MainWindow):
    """My UI"""
    def __init__(self, parent=None):
        super(MW, self).__init__(parent)
        self.setupUi(self)
        # Sets up the list widgets with properly formatted csv's
        pathdict = {}
        for d in os.listdir("./Materials/"):
            pathdict[d]=[]
            for file in os.listdir("./Materials/"+d):
                split = os.path.splitext(file)
                if split[1] == ".csv":
                    pathdict[d].append(split[0])
        self.lwDielectric.addItems(pathdict['Dielectric'])
        self.lwMetal.addItems(pathdict['Metal'])
        self.lwOther.addItems(pathdict['Other'])
        self.lWSemiconductors.addItems(pathdict['Semiconductor'])

        self.widget = MPLibWidget(self.GraphFrame)
        self.verticalLayout.addWidget(self.widget)
        self.btnAddLayer.clicked.connect(self.add_layer)
        self.btnRemoveLayer.clicked.connect(self.remove_layer)
        self.btnSwapLayer.clicked.connect(self.swap_layer)
        self.btnPlot.clicked.connect(self.plot_clicked)
        self.actionOpen.triggered.connect(self.open_data)
        self.btnFit.clicked.connect(self.fit)

        self.model = Model()
        self.data = None

    def swap_layer(self):
        """
        swaps the layer selected in the table widget with the selected layer in the list widget
        :return: None
        """
        selected = self.tableWidget.currentRow()
        ctab = self.tab_Library.currentIndex()
        name = self.tab_Library.tabText(ctab)
        list_widget = self.tab_Library.widget(ctab).findChild(QtWidgets.QListWidget)
        item = list_widget.currentItem()
        if item is not None:
            txt = item.text()
            if txt not in self.model.mat_df:
                self.model.add_material(txt, './Materials/' + name + '/' + txt + '.csv')
            txt = QtWidgets.QTableWidgetItem(txt)
            self.tableWidget.setItem(selected, 1, txt)

    def open_data(self):
        """
        opens a csv file, then runs a sav golay on it
        assigns data to the widget, then plots the sav'd plot
        :return:
        """
        root = tk.Tk()
        root.withdraw()
        root.lift()
        filename = askopenfilename(initialdir='C:/WORK!!!!/VF088 - ARC Thickness/')

        self.data = Data(filename, 'data', self.model.wavelength, 'R')
        self.widget.axes.cla()
        plot = self.widget.axes.plot(np.array(self.data.series.index), np.array(self.data.series), 'b-')
        self.widget.canvas.draw()

    def fit(self):
        """
        just a UI call for model.fit()
        currently always adds BK7 and SLG materials because it's hard coded
        :return:
        """
        if self.data is None:
            print("No Data to fit")
        else:
            self.model.add_material('BK7', './Materials/Dielectric/BK7.csv')
            self.model.add_material('SLG', './Materials/Dielectric/SLG.csv')
            self.widget.axes.cla()
            self.model.fit(self.data, self.widget)

    def add_layer(self):
        """
        Adds a new layer on top of current stack
        :return: None
        """
        current_tab = self.tab_Library.currentIndex()
        mat_type = self.tab_Library.tabText(current_tab)
        list_widget = self.tab_Library.widget(current_tab).findChild(QtWidgets.QListWidget)
        item = list_widget.currentItem()

        if item is not None:
            self.tableWidget.insertRow(0)
            rows = self.tableWidget.rowCount()-1
            rows = QtWidgets.QTableWidgetItem(str(rows))
            self.tableWidget.setItem(0, 0, rows)
            txt = item.text()
            if txt not in self.model.mat_df:
                self.model.add_material(txt, './Materials/' + mat_type + '/' + txt + '.csv')
            txt = QtWidgets.QTableWidgetItem(txt)
            self.tableWidget.setItem(0, 1, txt)
            self.tableWidget.setItem(0, 2, QtWidgets.QTableWidgetItem(str(100)))

    def remove_layer(self):
        """
        Deletes a layer
        :return: None
        """
        rows = self.tableWidget.rowCount()-1
        #  selected = self.tableWidget.selectionModel().selectedRows()
        selected = self.tableWidget.currentRow()
        if selected != rows:
            self.tableWidget.removeRow(selected)

    def plot_clicked(self):
        """
        Plots based on currently selected stack
        :return: None
        """
        layers = ['Air']
        d_list = [inf]

        rows = self.tableWidget.rowCount()
        for i in range(rows):
            layers.append(self.tableWidget.item(i, 1).text())
            d_list.append(int(self.tableWidget.item(i, 2).text()))
        d_list[-1] = inf
        theta = float(self.le_Theta0.text())*sp.pi/180
        self.model.layers = layers
        self.model.run(d_list, theta)
        r = self.model.data['R']
        t = self.model.data['T']
        a = 1-t-r
        self.widget.axes.cla()
        plots = self.widget.axes.plot(self.model.wavelength, r, self.model.wavelength, t, self.model.wavelength, a)
        self.widget.figure.legend(plots, ['R', 'T', 'A'])
        self.widget.canvas.draw()


class Material:
    """
    Each csv read gives a new DataFrame with n vs wavelength
    """
    def __init__(self, path, name, wavelengths):
        f = pd.read_csv(path)
        wv_raw = np.array(f.wv)
        nc = np.array(f.n+f.k*1j)  # complex refractive index
        self.name = name
        self.f = interp1d(wv_raw, nc, kind='cubic')
        self.min_wv = min(wv_raw)
        self.max_wv = max(wv_raw)
        self.df = pd.DataFrame(nc, wv_raw, [name])
        # self.df = self.df.reindex(wavelengths)
        # self.df = self.df.interpolate('spline', order=3)
        self.df = self.interp(wavelengths)

    def interp(self, wavelengths):
        """
        uses the self.f interpolation to reindex the dataframe
        :param wavelengths: np.array of wavelengths
        :return: a new dataframe
        """
        wavelengths = wavelengths[np.nonzero(np.logical_and(wavelengths > self.min_wv, wavelengths < self.max_wv))]
        nc = self.f(wavelengths)
        return pd.DataFrame(nc, wavelengths, [self.name])


class Data:
    def __init__(self, path, name, wavelengths, c_type):
        f = pd.read_csv(path)
        wv_raw = np.array(f.wv)
        R = np.array(f.R)
        self.name = name
        self.f = interp1d(wv_raw, R, kind='cubic')
        self.min_wv = min(wv_raw)
        self.max_wv = max(wv_raw)
        self.raw_series = pd.Series(data=R, index=wv_raw, name=name)
        self.interp(wavelengths)
        self.series = self.norm(wavelengths)
        self.weight_amp = 100
        self.weight_wid = 30
        self.weight_cen = 597
        self.weight = self.get_weighting(self.weight_cen, self.weight_amp, self.weight_wid)

    def interp(self, wavelengths):
        """
        :param wavelengths:list or array of wavelengths to interpolate over
        :return: should probably make this return instead of reassign
        """
        wavelengths = wavelengths[np.nonzero(np.logical_and(wavelengths > self.min_wv, wavelengths < self.max_wv))]
        data = self.f(wavelengths)
        self.raw_series = pd.Series(data=data, index=wavelengths, name=self.name)

    def norm(self, wavelength, data_filter="Savitzky-Golay"):
        """
        normalizer for the data with the optional default Sav-Golay filter
        :param wavelength: list or array of wavelengths to us
        :param data_filter: raw if other than Savitzky-Golay,
            otherwise calls S-G
        :return:
        """
        s = self.raw_series.ix[wavelength]
        if data_filter == "Savitzky-Golay":
            arr = self.savitzky_golay(y=np.array(s), window_size=21, order=3, deriv=0, rate=1)
            s = pd.Series(arr, s.index, name=self.raw_series.name+' processed')
        s = s.dropna()
        s = s - s.min()
        s = s / s.max()
        return s

    def get_weighting(self, cen, amp, wid):
        x = np.array(self.series.index)
        g = amp * np.exp(-(x-cen)**2 /wid)
        return pd.Series(1+g, x, name="Weighting")

    def savitzky_golay(self, y, window_size, order, deriv=0, rate=1):
        r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,)
            the values of the time history of the signal.
        window_size : int
            the length of the window. Must be an odd integer number.
        order : int
            the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        deriv: int
            the order of the derivative to compute (default = 0 means only smoothing)
        rate: int
            don't know
        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
           Data by Simplified Least Squares Procedures. Analytical
           Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
           W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
           Cambridge University Press ISBN-13: 9780521880688
        """

        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode='valid')


class Model:
    """
    Model using modification of SJByrnes tmm to allow simultaeous solution of multiple wavelengths
    """
    def __init__(self):
        self.wavelength = np.arange(300, 1700, 1)
        self.increment = .2
        self.index_array = np.array([])
        self.materials = []
        self.mat_df = pd.DataFrame(1+0j, self.wavelength, ['Air'])
        self.data = {}
        self.add_material("TCO", './Materials/Semiconductor/TCO.csv')
        self.layers = []

    def add_material(self, film, path):
        """
        Puts a material into the mat_df dataframe
        :param film: Name of the film, this should always be the name of the csv
        :param path: path to the CSV
        :return:
        """
        if film not in self.mat_df:
            mat = Material(path, film, self.wavelength)
            self.materials.append(mat)
            self.mat_df = self.mat_df.join(mat.df)

    def bruggeman(self, n1, n2, percent_included):
        """
        Bruggeman Effective Medium Approximation
        :param n1: Bulk material complex index
        :param n2: Included material complex index
        :param percent_included: 0-1
        :return: dict of dielectric, index, and inputs
        """
        p = n1/n2
        b = .25*((3*percent_included-1)*(1/p-p)+p)
        z = b + (b**2 + .5)**0.5
        e = z*n1*n2
        return {"e": e, "n": e**0.5, 'conc': percent_included, "n1": n1, 'n2': n2}

    def brug_transform(self, df, layer, incl, percent):
        """
        :param df: a given dataframe that you want to transform one of the layers
        :param layer: which layer you want to transform
        :param incl: included material complex index
        :param percent: 0-1
        :return: void
        """
        p = df[layer]/incl
        b = .25*((3*percent-1)*(1/p-p)+p)
        z = b + (b**2 + .5)**0.5
        e = z*df[layer]*incl
        n = e**.5
        df[layer] = n

    def run(self, thicknesses, theta0, polarization=None):
        """
        runs the model with specified parameters
        :param thicknesses: list of thicknesses, first and last must be inf
        :param theta0: input angle
        :param polarization: polarization state 's', 'p', or None
        :return: void
        """
        self.index_array = np.array(self.mat_df[self.layers])
        if polarization is None:
            self.data = tmm.unpolarized_RT(self.index_array, thicknesses, theta0, self.wavelength)
        elif polarization in ['p', 's']:
            self.data = tmm.coh_tmm(polarization, self.index_array, thicknesses, theta0, self.wavelength)

    def get_R(self, wavelengths, thickness, theta, void_percent):
        thicknesses = [inf, thickness, inf]
        mat = self.mat_df.ix[wavelengths]
        mat = mat[self.layers]
        theta0 = theta*sp.pi/180
        self.brug_transform(mat, self.layers[1], mat['Air'], void_percent)
        self.index_array = np.array(mat)
        self.data = tmm.unpolarized_RT(self.index_array, thicknesses, theta0, wavelengths)
        R = self.data['R']
        R -= min(R)
        R /= max(R)
        return R

    def norm(self, wavelength):
        df = self.mat_df.ix[wavelength]
        df = df - df.min()
        df = df / df.max()
        return df

    def fit(self, data, widget):
        self.layers = ['Air', 'BK7', 'SLG']
        ax_limits = widget.axes.axis()

        x_min = round(max(ax_limits[0], data.series.index.min()))
        x_max = round(min(ax_limits[1], data.series.index.max()))
        wv = np.arange(x_min, x_max)

        mod = lmfit.Model(self.get_R, ['wavelengths'], ['thickness', 'theta', 'void_percent'])
        mod.set_param_hint('thickness', value=130, min=50, max=250)
        mod.set_param_hint('theta', value=45, min=44, max=46, vary=False)
        mod.set_param_hint('void_percent', value=.15, min=.05, max=.5)

        R = data.norm(wv, False)
        result = mod.fit(R, wavelengths=wv)

        RMSE = (sp.sum(result.residual**2)/(result.residual.size-2))**0.5
        bf_values = result.best_values
        bf_str = 'thk: ' + str(round(bf_values['thickness'])) + ", Void %: " + str(round(bf_values['void_percent']*100, 2))
        txt_spot = wv.min()-100 + (wv.max()-wv.min()) / 2

        ax = widget.figure.axes[0]
        result.plot_fit(ax=ax, datafmt='b+', initfmt='r--', fitfmt='g-')
        ax.text(txt_spot, .9, "RMSE: "+str(round(RMSE, 3)))
        ax.text(txt_spot, .85, bf_str)
        widget.canvas.draw()
        print(result.fit_report())

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = MW()
    form.show()
    app.exec_()
