from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import *
from tkinter import ttk, filedialog
import numpy as np
import io
import cv2
import matplotlib
from PIL import Image
from PIL import ImageTk
matplotlib.use("TkAgg")


class Widget(Frame):
    IMG = None
    TempIMG = None

    def __init__(self, parent=None):
        Frame.__init__(self, parent)
        self.parent = parent
        self.pack()
        self.make_widgets()

    def make_widgets(self):
        self.winfo_toplevel().title("GUI")
        self.winfo_toplevel().geometry("1200x600")

        self.LeftFrame = LabelFrame(
            self, text="Oryginal image", height=500, width=300)
        self.LeftFrame.grid(row=0, column=0, columnspan=5, rowspan=5)

        self.ImagePanel = Canvas(self.LeftFrame, height=500, width=300)
        self.ImagePanel.pack(expand=YES, fill=BOTH)
        self.ImageOnPanel = self.ImagePanel.create_image(0, 0, anchor=NW)

        self.Load = Button(self, text="Select an image",
                           command=self.select_image)
        self.Load.grid(row=6, column=0)

        self.RightFrame = LabelFrame(
            self, text="Modified image", height=500, width=300)
        self.RightFrame.grid(row=0, column=5, columnspan=5, rowspan=5)
        self.ImagePanel2 = Canvas(self.RightFrame, height=500, width=300)
        self.ImagePanel2.pack(expand=YES, fill=BOTH)
        self.ImageOnPanel2 = self.ImagePanel2.create_image(0, 0, anchor=NW)
        self.RightFrame = LabelFrame(
            self, text="Modified image", height=500, width=300)

        self.RightFrame.grid(row=0, column=5, columnspan=5, rowspan=5)
        self.ImagePanel2 = Canvas(self.RightFrame, height=500, width=300)
        self.ImagePanel2.pack(expand=YES, fill=BOTH)
        self.ImageOnPanel2 = self.ImagePanel2.create_image(0, 0, anchor=NW)

        self.Choose = ttk.Combobox(self, values=["Edges", "Binary Threshold"])
        self.Choose.current(0)
        self.Choose.grid(row=6, column=1, rowspan=3)

        self.Confirm = Button(self, text="Confirm", command=self.confirm)
        self.Confirm.grid(row=6, column=4)

        self.PlotFrame = LabelFrame(
            self, text="Plot", height=500, width=300)
        self.PlotFrame.grid(row=0, column=10, columnspan=5, rowspan=5)
        self.Fig = Figure()
        self.Plot = self.Fig.add_subplot(1, 1, 1)

        self.canvas = FigureCanvasTkAgg(self.Fig, self.PlotFrame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        self.Hist = Button(self, text="Calculate Histogram",
                           command=self.calc_hist)
        self.Hist.grid(row=6, column=11)

        self.Slider = Scale(self, from_=0, to=255, orient=HORIZONTAL)
        self.Slider.set(128)
        self.Slider.grid(row=6, column=5)

    def select_image(self):
        # filename = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(
        #     ("jpeg files", "*.jpg"),
        #     ("all files", "*.*")
        # ))
        filename = '/home/taranchik/Downloads/backlit.jpg'
        if len(filename) > 0:
            print(filename)
            tmp = cv2.imread(filename)
            self.IMG = IMG = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        self.show_pic()

    def show_pic(self):
        if self.IMG.size > 0:
            img = Image.fromarray(self.IMG)

            h = self.ImagePanel.winfo_height()
            w = self.ImagePanel.winfo_width()

            h_ratio = h/self.IMG.shape[0]
            w_ratio = w/self.IMG.shape[1]

            if (h_ratio < 1.0) | (w_ratio < 1.0):
                if h_ratio < w_ratio:
                    ratio = h_ratio*w/w_ratio
                    img = img.resize((round(ratio), round(h)), Image.ANTIALIAS)
                else:
                    ratio = w_ratio*h/h_ratio
                    img = img.resize((round(w), round(ratio)), Image.ANTIALIAS)

            self.ImagePanel.ImgCatch = ImageTk.PhotoImage(img)
            self.ImagePanel.itemconfigure(
                self.ImageOnPanel, image=self.ImagePanel.ImgCatch)

        if self.TempIMG.size > 0:
            img = Image.fromarray(self.TempIMG)

            h = self.ImagePanel.winfo_height()
            w = self.ImagePanel.winfo_width()

            h_ratio = h/self.TempIMG.shape[0]
            w_ratio = w/self.TempIMG.shape[1]

            if (h_ratio < 1.0) | (w_ratio < 1.0):
                if h_ratio < w_ratio:
                    ratio = h_ratio*w/w_ratio
                    img = img.resize((round(ratio), round(h)), Image.ANTIALIAS)
                else:
                    ratio = w_ratio*h/h_ratio
                    img = img.resize((round(w), round(ratio)), Image.ANTIALIAS)

            self.ImagePanel2.ImgCatch = ImageTk.PhotoImage(img)
            self.ImagePanel2.itemconfigure(
                self.ImageOnPanel2, image=self.ImagePanel2.ImgCatch)

        self.canvas.draw()

    def confirm(self):
        if(self.Choose.current() == 0):
            self.TempIMG = cv2.Canny(self.IMG, 50, 100)

        if(self.Choose.current() == 1):
            res, ret = cv2.threshold(
                cv2.cvtColor(self.IMG, cv2.COLOR_RGB2GRAY),
                self.Slider.get(), 255, cv2.THRESH_BINARY)
            self.TempIMG = ret

        self.show_pic()

    def calc_hist(self):
        if self.IMG.size > 0:
            self.Plot.cla()
            color = ('r', 'g', 'b')
            for i, col in enumerate(color):
                histr = cv2.calcHist([self.IMG], [i], None, [256], [0, 256])
                self.Plot.plot(histr, color=col)
        self.show_pic()


if __name__ == "__main__":
    root = Tk()
    something = Widget(root)
    root.mainloop()
