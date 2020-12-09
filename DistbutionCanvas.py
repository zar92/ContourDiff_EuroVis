from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure
import ProcessData

import seaborn as sns


class DistrbutionCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=80):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.subplots_adjust(bottom=0.3)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

    def dist_plot(self,file,column):
        data = ProcessData.importData(file)
        print('hi')
        data['levels'] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
        sns.despine(fig=self.fig, ax=self.axes,
                    top=True, right=True, left=True, bottom=True)
        plot = sns.distplot(data['levels'], ax=self.axes)

        plot.set_yticks([])

    def clearPlt(self):
        self.fig.clear()
        self.axes = self.figure.add_subplot(111)
        self.draw()
    def addVerticalLines(self,x1,x2,x3):
        self.axes.scatter([x1,x2,x3],[0,0,0],color="k")
        self.draw()