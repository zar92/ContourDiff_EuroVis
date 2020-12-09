from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

import ProcessData

import numpy as np
import skimage.io as io
import skimage.transform as transform


class Canvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=10, dpi=80):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

    def symlog(self, x, y):
        mapx = np.abs(x)
        mapy = np.abs(y)
        val = 5 * np.log2(np.sqrt(mapx ** 2 + mapy ** 2))
        val[val < 0] = 0
        mapx = val * np.cos(np.arctan(mapy / (mapx + .0000001)))
        mapy = val * np.sin(np.arctan(mapy / (mapx + .0000001)))
        return np.sign(x) * mapx, np.sign(y) * mapy

    def symexp(self, x, y):
        mapx = np.abs(x)
        mapy = np.abs(y)
        val = np.power(2, np.sqrt(mapx ** 2 + mapy ** 2))
        val[val < 0] = 0
        mapx = val * np.cos(np.arctan(mapy / (mapx + .0000001)))
        mapy = val * np.sin(np.arctan(mapy / (mapx + .0000001)))
        return np.sign(x) * mapx, np.sign(y) * mapy

    def spaghettiPlot(self):
        data = ProcessData.importData('data0.csv')['SMOIS']
        self.axes.contour(np.array(data).reshape(699, 639))
        self.draw()

    def filledContour(self):
        data = ProcessData.importData('data0.csv')['SMOIS']
        self.contourf = self.axes.contourf(np.array(data).reshape(699, 639))
        self.draw()

    def clearPlt(self):
        self.fig.clear()
        self.axes = self.figure.add_subplot(111)
        self.draw()

    def clearPlt2(self):
        self.fig.clear()
        self.axes = self.figure.add_subplot(111)

    def plot_contour(self,data,level):
        self.axes.contour(np.array(data['levels']).reshape(699, 639), level, colors=['g', 'r', 'y'])
        self.draw()

    # def generate_images(self,filtered_graph,data,levels,column,alpha_cf = 0.7,flag_dir = 0,flag_content = 0,magnitude = 0,
    #                     cmap='Colormap 1',cline='copper',arrowscale = 'Linear', cvector_high2low = 'Black', cvector_low2high = 'Blue', line_opacity = 0.4,line_width=1.5):

    def generate_images(self, filtered_graph, filename, data, levels, column, alpha_cf=1, flag_dir=0,
                            flag_content=0, magnitude=0,
                            cmap='Colormap 1', cline='copper', arrowscale='Linear', cvector_high2low='Black',
                            cvector_low2high='Blue', line_opacity=0.4, line_width=0.5):
        print ("hey")

        def getRGBdecr(hex):
            hex = hex.lstrip('#')
            hlen = len(hex)
            return tuple(np.array([int(hex[i:i + hlen // 3], 16) for i in range(0, hlen, hlen // 3)]) / 255.0)

        rgb_colors = [getRGBdecr('#d7191c'), getRGBdecr('#fdae61'), getRGBdecr('#a6d96a'), getRGBdecr('#1a9641')]

        cmap_dict = {
            'Colormap 1': ['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c'],
            'Colormap 2': ['#f6eff7', '#bdc9e1', '#67a9cf', '#02818a'],
            'Colormap 3': ['#238b45', '#66c2a4', '#b2e2e2', '#edf8fb'],
            'Colormap 4': ['#feebe2', '#fbb4b9', '#f768a1', '#ae017e'],
            'Colormap 5': ['#f1eef6', '#bdc9e1', '#74a9cf', '#0570b0'],
            'Colormap 6': ['#f2f0f7', '#cbc9e2', '#9e9ac8', '#6a51a3']
            # 'Colormap 1':['#7fc97f','#beaed4','#fdc086','#ffff99'],
            # 'Colormap 2':['#1b9e77','#d95f02','#7570b3','#e7298a'],
            # 'Colormap 3':['#a6cee3','#1f78b4','#b2df8a','#33a02c'],
            # 'Colormap 4':['#e41a1c','#377eb8','#4daf4a','#984ea3'],
            # 'Colormap 5' :['#66c2a5','#fc8d62','#8da0cb','#e78ac3'],
            # 'Colormap 6':['#8dd3c7','#ffffb3','#bebada','#fb8072']
        }

        self.clearPlt2()

        # filtered_graph = filtered_graph[['level', 'node_x', 'node_y', 'path', 'aggregated_weight', 'actual_weight', 'normalized', 'res_dir_x_pos','res_dir_x_neg',
        #      'res_dir_y_pos', 'res_dir_y_neg', 'res_dir_x_1_pos', 'res_dir_x_1_neg', 'res_dir_y_1_pos', 'res_dir_y_1_neg', 'resultant_pos', 'resultant_neg', 'mag_pos', 'mag_neg']]

        if filename[-3:] != 'csv':
            ext = self.axes.get_xlim() + self.axes.get_ylim()
            self.axes.imshow(transform.resize(io.imread(filename), (699, 639)))

        filtered_graph = filtered_graph[
            ['level', 'node_x', 'node_y', 'path', 'aggregated_weight', 'actual_weight', 'normalized',
             'res_dir_x_HL', 'res_dir_y_HL', 'res_dir_x_LH', 'res_dir_y_LH',
             'res_dir_x_1_HL', 'res_dir_y_1_HL', 'res_dir_x_1_LH', 'res_dir_y_1_LH',
             'resultant', 'mag_HL', 'mag_LH']]

        data['levels'] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())

        self.axes.contour(np.array(data['levels']).reshape(699, 639), levels, cmap=cline, alpha=line_opacity,linewidths=line_width)
        # self.axes.contour(np.array(data['levels']).reshape(699, 639), levels, colors=cmap_dict[cmap], alpha=line_opacity, linewidths=line_width)
        print ("hey2")
        if(flag_content == 0 or flag_content == 1):
            # self.axes.contourf(np.array(data['levels']).reshape(699, 639), levels, colors=cmap_dict[cmap], alpha=alpha_cf)
            self.axes.contourf(np.array(data['levels']).reshape(699, 639), [0, levels[0], levels[1], levels[2], 1], colors=cmap_dict[cmap], alpha=alpha_cf)

            colors= cmap_dict[cmap]
            texts = ["0 - Isoline1", "Isoline1 - Isoline2", "Isoline2 - Isoline3", "Isoline3 - 100"]
            patches = [mpatches.Patch(color=colors[i],label = "{:s}".format(texts[i]))for i in range(len(texts))]
            self.axes.legend(handles = patches, bbox_to_anchor = (0.5, 0.01), loc = 'lower center', ncol=4)

        # filtered_graph = filtered_graph[filtered_graph['normalized'] >= 0.01]
        # df1_pos = filtered_graph[(filtered_graph['resultant_pos'] > -math.inf) & (filtered_graph['mag_pos'] > magnitude)].copy()
        # df1_neg = filtered_graph[(filtered_graph['resultant_neg'] < math.inf) & (filtered_graph['mag_neg'] > magnitude)].copy()

        p10 = np.percentile(filtered_graph['normalized'], 10)
        p90 = np.percentile(filtered_graph['normalized'], 90)
        filtered_graph = filtered_graph[(filtered_graph['normalized'] >= p10) & (filtered_graph['normalized'] <= p90)]
        df1 = filtered_graph[  # (filtered_graph['resultant'] >= 0) &
            (filtered_graph['mag_HL'] >= magnitude)].copy()
        df2 = filtered_graph[  # (filtered_graph['resultant'] < 0) &
            (filtered_graph['mag_LH'] >= magnitude)].copy()

######## for visualizing simplified contourmap, code in the last part, will add later ###

# Exponential, Logarithmic and Normalized value finding
#         df1.loc[:, 'exp_x_1'] = np.exp(df1['res_dir_x_1'])
#         df1.loc[:, 'exp_y_1'] = np.exp(df1['res_dir_y_1'])
#         df1.loc[:, 'log_x_1'] = (np.log(df1['res_dir_x_1'])).values
#         df1.loc[:, 'log_y_1'] = (np.log(df1['res_dir_y_1'])).values

#         x1 = df1['res_dir_x_1'].min()
#         y1 = df1['res_dir_x_1'].max()
#         x2 = df1['res_dir_y_1'].min()
#         y2 = df1['res_dir_y_1'].max()
#         df1.loc[:, 'nor_x_1'] = ((df1['res_dir_x_1'] - x1)/(y1-x1)).values
#         df1.loc[:, 'nor_y_1'] = ((df1['res_dir_y_1'] - x2)/(y2-x2)).values

#
#         df2.loc[:, 'exp_x_1'] = np.exp(df2['res_dir_x_1'])
#         df2.loc[:, 'exp_y_1'] = np.exp(df2['res_dir_y_1'])
#         df2.loc[:, 'log_x_1'] = (np.log(df2['res_dir_x_1'])).values
#         df2.loc[:, 'log_y_1'] = (np.log(df2['res_dir_y_1'])).values

#         x3 = df2['res_dir_x_1'].min()
#         y3 = df2['res_dir_x_1'].max()
#         x4 = df2['res_dir_y_1'].min()
#         y4 = df2['res_dir_y_1'].max()
#         df2.loc[:, 'nor_x_1'] = ((df2['res_dir_x_1'] - x3) / (y3 - x3)).values
#         df2.loc[:, 'nor_y_1'] = ((df2['res_dir_y_1'] - x4) / (y4 - x4)).values

        x1 = df1['res_dir_x_1_HL'].min()
        y1 = df1['res_dir_y_1_HL'].min()
        x2 = max(df1['res_dir_x_1_HL'].max(), -x1)
        y2 = max(df1['res_dir_y_1_HL'].max(), -y1)
        delta = max(x2, y2) + 0.00000001
        # jyoti: not perfect, but works now
        df1.loc[:, 'nor_x_1_HL'] = 5 * ((df1['res_dir_x_1_HL']) / (delta)).values
        df1.loc[:, 'nor_y_1_HL'] = 5 * ((df1['res_dir_y_1_HL']) / (delta)).values

        x3 = df2['res_dir_x_1_LH'].min()
        y3 = df2['res_dir_y_1_LH'].min()
        x4 = max(df2['res_dir_x_1_LH'].max(), -x3)
        y4 = max(df2['res_dir_y_1_LH'].max(), -y3)
        delta = max(x4, y4) + 0.00000001
        # jyoti: not perfect, but works now
        df2.loc[:, 'nor_x_1_LH'] = 3 * ((df2['res_dir_x_1_LH']) / (delta)).values
        df2.loc[:, 'nor_y_1_LH'] = 3 * ((df2['res_dir_y_1_LH']) / (delta)).values

# scale = math.exp(filtered_graph['mag'].min()+20)

        # if(flag_content == 0 or flag_content == 2):
        #     print ("brits2")
        #     if(arrowscale == 'Linear'):
        #         if(flag_dir == 0 or flag_dir == 1):
        #             print("0 1 linear")
        #             self.axes.quiver(df1_pos['node_x'], df1_pos['node_y'], df1_pos['res_dir_x_1_pos'], df1_pos['res_dir_y_1_pos'],
        #                 width=0.002, headwidth=5.5, headlength=5.5, color = cvector_high2low, scale=10)
        #
        #         if (flag_dir == 0 or flag_dir == 2):
        #             self.axes.quiver(df1_neg['node_x'], df1_neg['node_y'], df1_neg['res_dir_x_1_neg'], df1_neg['res_dir_y_1_neg'],
        #                 width=0.002, headwidth=5.5, headlength=5.5, color = cvector_low2high, scale=10)
        #     elif(arrowscale == 'Exponential'):
        #         if (flag_dir == 0 or flag_dir == 1):
        #             print("0 1 exponential")
        #             self.axes.quiver(df1_pos['node_x'], df1_pos['node_y'], df1_pos['res_dir_x_1_pos'], df1_pos['res_dir_y_1_pos'],
        #                 width=0.002, headwidth=5.5, headlength=5.5, color=cvector_high2low, scale=200)
        #
        #         if (flag_dir == 0 or flag_dir == 2):
        #             self.axes.quiver(df1_neg['node_x'], df1_neg['node_y'], df1_neg['res_dir_x_1_neg'], df1_neg['res_dir_y_1_neg'],
        #                 width=0.002, headwidth=5.5, headlength=5.5, color=cvector_low2high, scale=200)
        #
        #     elif (arrowscale == 'Logarithmic'):
        #         if (flag_dir == 0 or flag_dir == 1):
        #             print("0 1 logarithmic")
        #             self.axes.quiver(df1_pos['node_x'], df1_pos['node_y'], df1_pos['res_dir_x_1_pos'], df1_pos['res_dir_y_1_pos'],
        #                 width=0.002, headwidth=5.5, headlength=5.5, color=cvector_high2low, scale=10000)
        #
        #         if (flag_dir == 0 or flag_dir == 2):
        #             self.axes.quiver(df1_neg['node_x'], df1_neg['node_y'], df1_neg['res_dir_x_1_neg'], df1_neg['res_dir_y_1_neg'],
        #                 width=0.002, headwidth=5.5, headlength=5.5, color=cvector_low2high, scale=10000)

            # elif (arrowscale == 'Normalized'):
            #     if (flag_dir == 0 or flag_dir == 1):
            #         print("0 1 Exponential")
            #         self.axes.quiver(df1['node_x'], df1['node_y'], df1['nor_x_1'], df1['nor_y_1'],
            #                          width=0.0009, headwidth=5.5, headlength=5.5, color=cvector_high2low, scale=100)
            #
            #     if (flag_dir == 0 or flag_dir == 2):
            #         self.axes.quiver(df2['node_x'], df2['node_y'], df2['nor_x_1'], df2['nor_y_1'],
            #                          width=0.0009, headwidth=5.5, headlength=5.5, color=cvector_low2high, scale=100)

        if (flag_content == 0 or flag_content == 2):
            if (arrowscale == 'Linear'):
                if (flag_dir == 0 or flag_dir == 1):
                    self.axes.quiver(df1['node_x'], df1['node_y'], df1['res_dir_x_1_HL'], df1['res_dir_y_1_HL'],
                                     width=0.0009, headwidth=5.5, headlength=5.5, color=cvector_high2low,
                                     scale_units='inches', scale=100)

                if (flag_dir == 0 or flag_dir == 2):
                    self.axes.quiver(df2['node_x'], df2['node_y'], df2['res_dir_x_1_LH'], df2['res_dir_y_1_LH'],
                                     width=0.0009, headwidth=5.5, headlength=5.5, color=cvector_low2high,
                                     scale_units='inches', scale=100)

            elif (arrowscale == 'Exponential'):
                if (flag_dir == 0 or flag_dir == 1):
                    a, b = self.symexp(df1['nor_x_1_HL'], df1['nor_y_1_HL'])
                    self.axes.quiver(df1['node_x'], df1['node_y'], a, b,
                                     width=0.0009, headwidth=5.5, headlength=5.5, color=cvector_high2low,
                                     scale_units='inches', scale=10)

                if (flag_dir == 0 or flag_dir == 2):
                    a, b = self.symexp(df2['nor_x_1_LH'], df2['nor_y_1_LH'])
                    self.axes.quiver(df2['node_x'], df2['node_y'], a, b,
                                     width=0.0009, headwidth=5.5, headlength=5.5, color=cvector_low2high,
                                     scale_units='inches', scale=10)

            elif (arrowscale == 'Logarithmic'):
                if (flag_dir == 0 or flag_dir == 1):
                    a, b = self.symlog(df1['nor_x_1_HL'], df1['nor_y_1_HL'])
                    self.axes.quiver(df1['node_x'], df1['node_y'], a, b,
                                     width=0.0009, headwidth=5.5, headlength=5.5, color=cvector_high2low,
                                     scale_units='inches', scale=10)

                if (flag_dir == 0 or flag_dir == 2):
                    a, b = self.symlog(df2['nor_x_1_LH'], df2['nor_y_1_LH'])
                    self.axes.quiver(df2['node_x'], df2['node_y'], a, b,
                                     width=0.0009, headwidth=5.5, headlength=5.5, color=cvector_low2high,
                                     scale_units='inches', scale=10)

            elif (arrowscale == 'Normalized'):
                if (flag_dir == 0 or flag_dir == 1):
                    self.axes.quiver(df1['node_x'], df1['node_y'], df1['nor_x_1_HL'], df1['nor_y_1_HL'],
                                     width=0.0009, headwidth=5.5, headlength=5.5, color=cvector_high2low,
                                     scale_units='inches', scale=10)

                if (flag_dir == 0 or flag_dir == 2):
                    self.axes.quiver(df2['node_x'], df2['node_y'], df2['nor_x_1_LH'], df2['nor_y_1_LH'],
                                     width=0.0009, headwidth=5.5, headlength=5.5, color=cvector_low2high,
                                     scale_units='inches', scale=10)

        self.draw()





# filtered_graph_level_0 = filtered_graph[(filtered_graph['level'] == 0)]
        # filtered_graph_level_0 = ProcessData.assignColor(filtered_graph_level_0, 'normalized')
        # max_path = filtered_graph_level_0['path'].max()
        # color_list = filtered_graph_level_0['color'].tolist()
        # start_time_for_creating_contour_paths = time.time()
        #
        # for i in range(max_path):
        #     current_path_x = filtered_graph_level_0[filtered_graph_level_0['path'] == i]['node_x']
        #     current_path_y = filtered_graph_level_0[filtered_graph_level_0['path'] == i]['node_y']
        #     if (len(current_path_x) > 15):
        #         points = np.array([current_path_x, current_path_y]).T
        #         distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        #         distance = np.insert(distance, 0, 0) / distance[-1]
        #         interpolator = interp1d(distance, points, kind='cubic', axis=0)
        #         alpha = np.linspace(0, 1, 100)
        #         current_path_x = pd.Series(interpolator(alpha).T[0])
        #         current_path_y = pd.Series(interpolator(alpha).T[1])
        #
        #     list_current_points = list(zip(current_path_x.tolist(), current_path_y.tolist()))
        #     if ((len(list_current_points) >= 3) & (not (current_path_y.eq(0).any())) & (
        #     (current_path_y < 600).all()) & ((current_path_x < 600).all())):
        #         color = filtered_graph_level_0[filtered_graph_level_0['path'] == i]['color'].tolist()[0]
        #         ring = LinearRing(list_current_points)
        #         x, y = ring.xy
        #         plt.plot(x, y, color=color)
        #     else:
        #         if (len(filtered_graph_level_0[filtered_graph_level_0['path'] == i]['color'].tolist()) > 0):
        #             color = filtered_graph_level_0[filtered_graph_level_0['path'] == i]['color'].tolist()[0]
        #             plt.plot(current_path_x, current_path_y, 'C3', lw=1, color=color)
        #
        # filtered_graph_level_1 = filtered_graph[(filtered_graph['level'] == 1)]
        # filtered_graph_level_1 = ProcessData.assignColor(filtered_graph_level_1, 'normalized')
        # max_path = filtered_graph_level_1['path'].max()
        # color_list = filtered_graph_level_1['color'].tolist()
        # for i in range(max_path):
        #     current_path_x = filtered_graph_level_1[filtered_graph_level_1['path'] == i]['node_x']
        #     current_path_y = filtered_graph_level_1[filtered_graph_level_1['path'] == i]['node_y']
        #     if (len(current_path_x) > 15):
        #         points = np.array([current_path_x, current_path_y]).T
        #         distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        #         distance = np.insert(distance, 0, 0) / distance[-1]
        #         interpolator = interp1d(distance, points, kind='cubic', axis=0)
        #         alpha = np.linspace(0, 1, 100)
        #         current_path_x = pd.Series(interpolator(alpha).T[0])
        #         current_path_y = pd.Series(interpolator(alpha).T[1])
        #
        #     list_current_points = list(zip(current_path_x.tolist(), current_path_y.tolist()))
        #     if ((len(list_current_points) >= 3) & (not (current_path_y.eq(0).any())) & (
        #     (current_path_y < 600).all()) & ((current_path_x < 600).all())):
        #         color = filtered_graph_level_1[filtered_graph_level_1['path'] == i]['color'].tolist()[0]
        #         ring = LinearRing(list_current_points)
        #         x, y = ring.xy
        #         plt.plot(x, y, color=color)
        #     else:
        #         if (len(filtered_graph_level_1[filtered_graph_level_1['path'] == i]['color'].tolist()) > 0):
        #             color = filtered_graph_level_1[filtered_graph_level_1['path'] == i]['color'].tolist()[0]
        #             plt.plot(current_path_x, current_path_y, 'C3', lw=1, color=color)
        #
        # filtered_graph_level_2 = filtered_graph[(filtered_graph['level'] == 2)]
        # filtered_graph_level_2 = ProcessData.assignColor(filtered_graph_level_2, 'normalized')
        # max_path = filtered_graph_level_2['path'].max()
        # color_list = filtered_graph_level_2['color'].tolist()
        # for i in range(max_path):
        #     current_path_x = filtered_graph_level_2[filtered_graph_level_2['path'] == i]['node_x']
        #     current_path_y = filtered_graph_level_2[filtered_graph_level_2['path'] == i]['node_y']
        #     if (len(current_path_x) > 15):
        #         points = np.array([current_path_x, current_path_y]).T
        #         distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        #         distance = np.insert(distance, 0, 0) / distance[-1]
        #         interpolator = interp1d(distance, points, kind='cubic', axis=0)
        #         alpha = np.linspace(0, 1, 100)
        #         current_path_x = pd.Series(interpolator(alpha).T[0])
        #         current_path_y = pd.Series(interpolator(alpha).T[1])
        #
        #     list_current_points = list(zip(current_path_x.tolist(), current_path_y.tolist()))
        #     if ((len(list_current_points) >= 3) & (not (current_path_y.eq(0).any())) & (
        #     (current_path_y < 600).all()) & ((current_path_x < 600).all())):
        #         color = filtered_graph_level_2[filtered_graph_level_2['path'] == i]['color'].tolist()[0]
        #         ring = LinearRing(list_current_points)
        #         x, y = ring.xy
        #         plt.plot(x, y, color=color)
        #     else:
        #         if (len(filtered_graph_level_2[filtered_graph_level_2['path'] == i]['color'].tolist()) > 0):
        #             color = filtered_graph_level_2[filtered_graph_level_2['path'] == i]['color'].tolist()[0]
        #             plt.plot(current_path_x, current_path_y, 'C3', lw=1, color=color)

        # df1 = filtered_graph[(filtered_graph['resultant'] >= 0) & (filtered_graph['mag'] > magnitude)].copy()
        # df2 = filtered_graph[(filtered_graph['resultant'] < 0) & (filtered_graph['mag'] > magnitude)].copy()