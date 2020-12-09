import numpy as np
import pandas as pd

from QuadTree import QTree
import time

from skimage import color
import skimage.io as io
import skimage.transform as transform
import skimage.filters as filt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops


def get_rgb_image(image):
    if len(image.shape) > 2 and image.shape[2] == 4:
        return color.rgba2rgb(image)

    return image


def extract_dataframe_from_img(filename):
    image = io.imread(filename)
    image = transform.resize(image, (699, 639))

    #excess_red channel
    # img_excess_red= 2 * image[:, :, 0] - image[:, :, 1] - image[:, :, 2]
    img_excess_red = image[:, :, 0]

    #otsu_thresholding
    img_filt = img_excess_red < filt.threshold_otsu(img_excess_red)

    # watershed modelling
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(img_filt)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=img_filt)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=img_filt)

    #labelling to make the background zero and keep the foreground only.
    pos = np.where(img_filt == False)
    lb = labels[pos[0][0]][pos[1][0]]
    labels[labels == labels[0][0]] = -1
    labels[labels == lb] = -1
    labels[labels != -1] = 1
    labels[labels == -1] = 0

    #biggest polygon capturing
    label_img = label(labels)
    regions = regionprops(label_img)
    prop = None
    mn = 0
    for props in regions:
        if props.area > mn:
            mn = props.area
            prop = props
    label_img[label_img != prop.label] = 0
    image[label_img <= 0, 0] = 0
    image[label_img <= 0, 1] = 0
    image[label_img <= 0, 2] = 0

    if len(image.shape) > 2:
        image = color.rgb2gray(get_rgb_image(image))
    rows = []
    for i in range(699):
        for j in range(639):
            rows.append([i, j, image[i][j]])
    df = pd.DataFrame(rows, columns=['latitude', 'longitude', 'gray_value'])
    return df


def importData(filename):
    """Read data from file and transform it into dataframe"""
    if filename[-3:] != 'csv':
        imagein = color.rgb2gray(get_rgb_image(io.imread(filename)))
        improcessed = transform.resize(imagein, (699, 639))
        rows = []
        for i in range(699):
            for j in range(639):
                rows.append([i, j, improcessed[i][j]])
        data = pd.DataFrame(rows, columns=['latitude', 'longitude', 'gray_value'])
    else:
        data = pd.read_csv(filename)

    return data

def createQuantile(data, column_name, cut_of_point):
    """Create quantile based on the cut of point(0.25,0.5,0.75)"""
    data[data[column_name] > data[column_name].quantile(cut_of_point)] = 0
    return data

def modelTheGraph(contourset):
    """Model the graph as dataframe from contourset"""
    cntr_data = pd.DataFrame(columns=['level', 'node_x', 'node_y', 'path'])
    frames = list()
    start_time_model_graph = time.time()

    for level_index in range(len(contourset.collections)):
        path_counter = 0
        indices = np.arange(0, len(contourset.collections[level_index].get_paths()))
        array_list = np.take(contourset.collections[level_index].get_paths(), indices)
        for item in array_list.flat:
            node_x = item.vertices[:, 0].tolist()
            node_y = item.vertices[:, 1].tolist()
            frames.append([level_index, node_x, node_y, path_counter])
            path_counter += 1
    df = pd.DataFrame(frames, columns=['level', 'node_x', 'node_y', 'path'])
    df1 = df[['level', 'node_y', 'path']]
    df2 = df[['level', 'node_x', 'path']]
    lst_col = 'node_y'
    r1 = pd.DataFrame({
        col: np.repeat(df1[col].values, df1[lst_col].str.len())
        for col in df1.columns.drop(lst_col)}
    ).assign(**{lst_col: np.concatenate(df1[lst_col].values)})[df1.columns]
    lst_col2 = 'node_x'
    r2 = pd.DataFrame({
        col: np.repeat(df2[col].values, df2[lst_col2].str.len())
        for col in df2.columns.drop(lst_col2)}
    ).assign(**{lst_col2: np.concatenate(df2[lst_col2].values)})[df2.columns]
    cntr_data['level'] = r1['level'].tolist()
    cntr_data['node_x'] = r2['node_x'].tolist()
    cntr_data['node_y'] = r1['node_y'].tolist()
    cntr_data['path'] = r1['path'].tolist()
    print("For modeling the graph %s seconds" % (time.time() - start_time_model_graph))
    return cntr_data


def dir_mag_by_5(filelist, column_name):
    """Calculate scalar diffrence of an entry againist its 24 neigbors"""
    mag_list = list()
    dir_list = list()
    for i in range(len(filelist)):
        padded_matrix_1 = np.pad(importData(filelist[i])[column_name].values.reshape(699, 639), [(1, 1), (1, 1)],
                                 mode='constant', constant_values=0)  # to calculate 1st degree neighbors
        padded_matrix_2 = np.pad(importData(filelist[i])[column_name].values.reshape(699, 639), [(2, 2), (2, 2)],
                                 mode='constant', constant_values=0)  # to calculate 2nd degree neighbors
        direction = np.zeros((699, 639, 24))
        org = padded_matrix_1[1:-1, 1:-1]

        # identifying the points
        dirs = {'dir_0': padded_matrix_1[1:-1, 2:], 'dir_1': padded_matrix_1[0:-2, 2:],
                'dir_2': padded_matrix_1[0:-2, 1:-1],
                'dir_3': padded_matrix_1[0:-2, 0:-2], 'dir_4': padded_matrix_1[1:-1, 0:-2],
                'dir_5': padded_matrix_1[2:, 0:-2],
                'dir_6': padded_matrix_1[2:, 1:-1], 'dir_7': padded_matrix_1[2:, 2:],
                'dir_8': padded_matrix_2[2:-2, 4:], 'dir_9': padded_matrix_2[0:-4, 4:],
                'dir_10': padded_matrix_2[0:-4, 2:-2],
                'dir_11': padded_matrix_2[0:-4, 0:-4], 'dir_12': padded_matrix_2[2:-2, 0:-4],
                'dir_13': padded_matrix_2[4:, 0:-4],
                'dir_14': padded_matrix_2[4:, 2:-2], 'dir_15': padded_matrix_2[4:, 4:],
                'dir_23': padded_matrix_2[3:-1, 4:],
                'dir_22': padded_matrix_2[4:, 3:-1], 'dir_21': padded_matrix_2[4:, 1:-3],
                'dir_20': padded_matrix_2[3:-1, 0:-4], 'dir_19': padded_matrix_2[1:-3, 0:-4],
                'dir_18': padded_matrix_2[0:-4, 1:-3], 'dir_17': padded_matrix_2[0:-4, 3:-1],
                'dir_16': padded_matrix_2[1:-3, 4:]
                }
        start_time_dir_mag = time.time()
        # calculating the differences
        for d in range(24):
            direction[:, :, d] = (org - dirs['dir_' + str(d)])

            # magnitude and direction list
            mag_list.append(np.linalg.norm(direction, axis=2))
            dir_list.append(direction)
        print("For computing direction and magnitude %s seconds" % (time.time() - start_time_dir_mag))
        return mag_list, dir_list

def draw_dirs2(filelist, column_name):
    """Calculate resultant direction in x and y"""
    start_time_vector = time.time()
    mag, direction = dir_mag_by_5(filelist, column_name)
    res_dir_x_list = list()
    res_dir_y_list = list()
    res_dir_x_list_HL = list()
    res_dir_y_list_HL = list()
    res_dir_x_list_LH = list()
    res_dir_y_list_LH = list()
    for i in range(len(filelist)):
        # for each file a structure is created to keep their directions
        res_dir_x = np.zeros_like(mag[i])
        res_dir_y = np.zeros_like(mag[i])

        # for each file, direction values in all directions are added
        for d in range(24):
            dir_neg = np.copy(direction[i][:, :, d])
            dir_neg[dir_neg > 0] = 0
            dir_neg = np.abs(dir_neg)

            if (d == 0 or d == 8):
                res_dir_x += dir_neg * np.cos(0)
                res_dir_y += dir_neg * np.sin(0)
            elif (d == 2 or d == 10):
                res_dir_x += dir_neg * np.cos(np.pi / 2)
                res_dir_y += -dir_neg * np.sin(np.pi / 2)
            elif (d == 4 or d == 12):
                res_dir_x += -dir_neg * np.cos(0)
                res_dir_y += dir_neg * np.sin(0)
            elif (d == 6 or d == 14):
                res_dir_x += dir_neg * np.cos(np.pi / 2)
                res_dir_y += dir_neg * np.sin(np.pi / 2)

            elif (d == 1 or d == 9):
                res_dir_x += dir_neg * np.cos(np.pi / 4)
                res_dir_y += -dir_neg * np.sin(np.pi / 4)
            elif (d == 3 or d == 11):
                res_dir_x += -dir_neg * np.cos(np.pi / 4)
                res_dir_y += -dir_neg * np.sin(np.pi / 4)
            elif (d == 5 or d == 13):
                res_dir_x += -dir_neg * np.cos(np.pi / 4)
                res_dir_y += dir_neg * np.sin(np.pi / 4)
            elif (d == 7 or d == 15):
                res_dir_x += dir_neg * np.cos(np.pi / 4)
                res_dir_y += dir_neg * np.sin(np.pi / 4)


            elif (d == 16):
                res_dir_x += dir_neg * np.cos(np.pi / 8)
                res_dir_y += -dir_neg * np.sin(np.pi / 8)
            elif (d == 17):
                res_dir_x += dir_neg * np.cos(3 * np.pi / 8)
                res_dir_y += -dir_neg * np.sin(3 * np.pi / 8)
            elif (d == 19):
                res_dir_x += -dir_neg * np.cos(np.pi / 8)
                res_dir_y += -dir_neg * np.sin(np.pi / 8)
            elif (d == 18):
                res_dir_x += -dir_neg * np.cos(3 * np.pi / 8)
                res_dir_y += -dir_neg * np.sin(3 * np.pi / 8)

            elif (d == 20):
                res_dir_x += -dir_neg * np.cos(np.pi / 8)
                res_dir_y += dir_neg * np.sin(np.pi / 8)

            elif (d == 21):
                res_dir_x += -dir_neg * np.cos(3 * np.pi / 8)
                res_dir_y += dir_neg * np.sin(3 * np.pi / 8)

            elif (d == 23):
                res_dir_x += dir_neg * np.cos(np.pi / 8)
                res_dir_y += dir_neg * np.sin(np.pi / 8)
            elif (d == 22):
                res_dir_x += dir_neg * np.cos(3 * np.pi / 8)
                res_dir_y += dir_neg * np.sin(3 * np.pi / 8)

            res_dir_x_list_LH.append(res_dir_x)
            res_dir_y_list_LH.append(res_dir_y)

        #################### NEG

        res_dir_x = np.zeros_like(mag[i])
        res_dir_y = np.zeros_like(mag[i])

        # for each file, direction values in all directions are added
        for d in range(24):
            dir_pos = np.copy(direction[i][:, :, d])
            dir_pos[dir_pos < 0] = 0
            dir_pos = np.abs(dir_pos)

            if (d == 0 or d == 8):
                res_dir_x += dir_pos * np.cos(0)
                res_dir_y += dir_pos * np.sin(0)
            elif (d == 2 or d == 10):
                res_dir_x += dir_pos * np.cos(np.pi / 2)
                res_dir_y += -dir_pos * np.sin(np.pi / 2)
            elif (d == 4 or d == 12):
                res_dir_x += -dir_pos * np.cos(0)
                res_dir_y += dir_pos * np.sin(0)
            elif (d == 6 or d == 14):
                res_dir_x += dir_pos * np.cos(np.pi / 2)
                res_dir_y += dir_pos * np.sin(np.pi / 2)

            elif (d == 1 or d == 9):
                res_dir_x += dir_pos * np.cos(np.pi / 4)
                res_dir_y += -dir_pos * np.sin(np.pi / 4)
            elif (d == 3 or d == 11):
                res_dir_x += -dir_pos * np.cos(np.pi / 4)
                res_dir_y += -dir_pos * np.sin(np.pi / 4)
            elif (d == 5 or d == 13):
                res_dir_x += -dir_pos * np.cos(np.pi / 4)
                res_dir_y += dir_pos * np.sin(np.pi / 4)
            elif (d == 7 or d == 15):
                res_dir_x += dir_pos * np.cos(np.pi / 4)
                res_dir_y += dir_pos * np.sin(np.pi / 4)


            elif (d == 16):
                res_dir_x += dir_pos * np.cos(np.pi / 8)
                res_dir_y += -dir_pos * np.sin(np.pi / 8)
            elif (d == 17):
                res_dir_x += dir_pos * np.cos(3 * np.pi / 8)
                res_dir_y += -dir_pos * np.sin(3 * np.pi / 8)
            elif (d == 19):
                res_dir_x += -dir_pos * np.cos(np.pi / 8)
                res_dir_y += -dir_pos * np.sin(np.pi / 8)
            elif (d == 18):
                res_dir_x += -dir_pos * np.cos(3 * np.pi / 8)
                res_dir_y += -dir_pos * np.sin(3 * np.pi / 8)

            elif (d == 20):
                res_dir_x += -dir_pos * np.cos(np.pi / 8)
                res_dir_y += dir_pos * np.sin(np.pi / 8)

            elif (d == 21):
                res_dir_x += -dir_pos * np.cos(3 * np.pi / 8)
                res_dir_y += dir_pos * np.sin(3 * np.pi / 8)

            elif (d == 23):
                res_dir_x += dir_pos * np.cos(np.pi / 8)
                res_dir_y += dir_pos * np.sin(np.pi / 8)
            elif (d == 22):
                res_dir_x += dir_pos * np.cos(3 * np.pi / 8)
                res_dir_y += dir_pos * np.sin(3 * np.pi / 8)

            res_dir_x_list_HL.append(res_dir_x)
            res_dir_y_list_HL.append(res_dir_y)

    print("For vector difference %s seconds" % (time.time() - start_time_vector))
    return res_dir_x_list_HL, res_dir_y_list_HL, res_dir_x_list_LH, res_dir_y_list_LH, mag, direction


def fetch_direction(file_list, column_name):
    """Aggregate resultant direction in"""
    start_time_aggrigating_vector_components = time.time()
    res_dir_x_list_HL, res_dir_y_list_HL, res_dir_x_list_LH, res_dir_y_list_LH, mag, direction = draw_dirs2(file_list, column_name)
    # creating structure like mag
    all_in_x_HL = np.zeros_like(mag[0])
    all_in_y_HL = np.zeros_like(mag[0])
    all_in_x_LH = np.zeros_like(mag[0])
    all_in_y_LH = np.zeros_like(mag[0])
    all_mag = np.zeros_like(mag[0])
    # aggregating all direction values in x and y positions for all files
    for i in range(len(res_dir_x_list_HL)):
        all_in_x_HL = np.add(all_in_x_HL, res_dir_x_list_HL[i]) * np.var(res_dir_x_list_HL[i])
        all_in_y_HL = np.add(all_in_y_HL, res_dir_y_list_HL[i]) * np.var(res_dir_y_list_HL[i])
    for i in range(len(res_dir_x_list_LH)):
        all_in_x_LH = np.add(all_in_x_LH, res_dir_x_list_LH[i]) * np.var(res_dir_x_list_LH[i])
        all_in_y_LH = np.add(all_in_y_LH, res_dir_y_list_LH[i]) * np.var(res_dir_y_list_LH[i])

    # creating 1 D array of aggregated results in X and Y directions
    res_x_HL = all_in_x_HL.ravel()
    res_y_HL = all_in_y_HL.ravel()

    res_x_LH = all_in_x_LH.ravel()
    res_y_LH = all_in_y_LH.ravel()

    # adding a column as aggregated recult in dataframe from a dataset
    data = importData(file_list[-1])
    # data = importData(file_list[0])
    data['res_x_HL'] = res_x_HL
    data['res_y_HL'] = res_y_HL
    data['res_x_LH'] = res_x_LH
    data['res_y_LH'] = res_y_LH

    print("For timestamp and aggregating vector components %s seconds" % (time.time() - start_time_aggrigating_vector_components))
    return data


def createWeightedGraph(contourdf, file_list, column_name):
    """Created a weighted graph from extracted contour"""
    # The function computes the length of each path. Which is then taken as a weight. Then, aggregated direction is taken for each point on the graph using fetchDirection() function.
    # Then, the resultant direction got multiplied with the weight of each path which gave us the final resultant along x and y axis.
    start_time_creating_weighted_graph = time.time()
    weights = np.full((len(contourdf)), 1)  # initialize weights to one
    contourdf['weights'] = weights

    # group the dataframe to count path_length(number of nodes in the path)
    path_length_df = contourdf.groupby(['level', 'path']).size().reset_index(name='path_length')

    # find all paths in all levels that have only one node or path length 1
    path_length_1_df = path_length_df[path_length_df['path_length'] == 1]
    cntr_data_weight_0 = contourdf[(np.isin(contourdf['level'], path_length_1_df['level'])) &
                                   (np.isin(contourdf['path'], path_length_1_df['path']))]

    # these path length 1 paths have weight 0
    pd.options.mode.chained_assignment = None  # default='warn'
    cntr_data_weight_0['weights'] = 0
    # finding all other paths
    cntr_data__weight_1 = contourdf[~(np.isin(contourdf['level'], path_length_1_df['level'])) |
                                    ~(np.isin(contourdf['path'], path_length_1_df['path']))]
    cntr_data_weight_1_diffrence = (cntr_data__weight_1.shift() - cntr_data__weight_1)
    #calculating weight as Sqr_root(x^2 + y^2); x = x2-x1, y = y2-y1
    cntr_data_weight_1_diffrence['calculated_weight'] = (np.sqrt(
        (cntr_data_weight_1_diffrence['node_x'].values) ** 2 + (
            cntr_data_weight_1_diffrence['node_y'].values) ** 2).tolist())
    cntr_data__weight_1['calculated_weight'] = cntr_data_weight_1_diffrence['calculated_weight'].tolist()
    cntr_data__weight_1['path_diff'] = cntr_data_weight_1_diffrence['path'].tolist()
    weight_list = cntr_data__weight_1['calculated_weight'].tolist()

    indices = cntr_data__weight_1.loc[cntr_data__weight_1['path_diff'] != 0]
    for index, row in indices.iterrows():
        if (len(weight_list) > index + 1):
            weight_list[index] = weight_list[index + 1]

    cntr_data__weight_1['act2'] = weight_list
    cntr_data__weight_1['actual_weight'] = weight_list
    cntr_data__weight_1 = cntr_data__weight_1[['level', 'node_x', 'node_y', 'path', 'actual_weight']]
    cntr_data_weight_0['actual_weight'] = cntr_data_weight_0['weights']
    cntr_data_weight_0 = cntr_data_weight_0[['level', 'node_x', 'node_y', 'path', 'actual_weight']]
    weighted_df = pd.concat([cntr_data_weight_0, cntr_data__weight_1])
    weighted_df = weighted_df.sort_values(['level', 'path'])
    weighted_df['aggregated_weight'] = weighted_df.groupby(['level', 'path'])['actual_weight'].transform('sum')
    # weighted_df['aggregated_weight'] = weighted_df.groupby(['level', 'path', 'node_x', 'node_y'])['actual_weight'].transform('sum')

    weighted_df = weighted_df[['level', 'node_x', 'node_y', 'path', 'aggregated_weight', 'actual_weight']]
    weighted_df['normalized'] = (weighted_df['aggregated_weight'] - weighted_df['aggregated_weight'].min()) / ((
                weighted_df['aggregated_weight'].max() - weighted_df['aggregated_weight'].min())+.00000001)
    print("For creating a weighted graph %s seconds" % (time.time() - start_time_creating_weighted_graph))
    final_vector = time.time()
    # fetching the direction values

    data = fetch_direction(file_list, column_name)
    data['node_x_1'] = data['longitude']
    data['node_y_1'] = data['latitude']
    # weighted_df['node_x_1'] = weighted_df['node_x'] // 1
    # weighted_df['node_y_1'] = weighted_df['node_y'] // 1
    # merged_df = weighted_df.merge(data, how='left')

    truecopy = weighted_df.copy()
    shiftedcopy = weighted_df.copy()
    truecopy['node_x_1'] = truecopy['node_x'] // 1
    truecopy['node_y_1'] = truecopy['node_y'] // 1
    shiftedcopy['node_x_1'] = (1 + shiftedcopy['node_x']) // 1
    shiftedcopy['node_y_1'] = (1 + shiftedcopy['node_y']) // 1
    weighted_df = pd.concat([truecopy, shiftedcopy])
    weighted_df.drop_duplicates(subset=['node_x_1', 'node_y_1'])
    merged_df = weighted_df.merge(data, how='left', on=['node_x_1', 'node_y_1'])

    # truecopy = weighted_df.copy()
    # shiftedcopy = weighted_df.copy()
    # truecopy['node_x_1'] = np.floor(truecopy['node_x'])
    # truecopy['node_y_1'] = np.floor(truecopy['node_y'])
    # # shiftedcopy['node_x_1'] = (1 + shiftedcopy['node_x']) // 1
    # # shiftedcopy['node_y_1'] = (1 + shiftedcopy['node_y']) // 1
    # # weighted_df = pd.concat([truecopy, shiftedcopy])
    # weighted_df = truecopy
    # weighted_df.drop_duplicates(subset=['node_x_1', 'node_y_1'])
    # merged_df = weighted_df.merge(data, how='left', on=['node_x_1', 'node_y_1'])

    # merged_df = merged_df[['res_x_pos', 'res_x_neg', 'res_y_pos', 'res_y_neg', 'node_x_1', 'node_y_1']]
    merged_df = merged_df[['res_x_HL', 'res_y_HL', 'res_x_LH', 'res_y_LH', 'node_x_1', 'node_y_1']]

    weighted_df['res_dir_x_HL'] = merged_df['res_x_HL'].tolist()
    weighted_df['res_dir_y_HL'] = merged_df['res_y_HL'].tolist()
    weighted_df['res_dir_x_LH'] = merged_df['res_x_LH'].tolist()
    weighted_df['res_dir_y_LH'] = merged_df['res_y_LH'].tolist()
    # jyoti
    weighted_df['res_dir_x_1_HL'] = weighted_df['res_dir_x_HL'] * weighted_df['actual_weight']
    weighted_df['res_dir_y_1_HL'] = weighted_df['res_dir_y_HL'] * weighted_df['actual_weight']
    weighted_df['res_dir_x_1_LH'] = weighted_df['res_dir_x_LH'] * weighted_df['actual_weight']
    weighted_df['res_dir_y_1_LH'] = weighted_df['res_dir_y_LH'] * weighted_df['actual_weight']

    # weighted_df['res_dir_x_1_HL'] = weighted_df.groupby(['level', 'path', 'node_x_1', 'node_y_1'])['res_dir_x_1_HL'].transform('sum') / (.0001 + weighted_df['aggregated_weight'])
    # weighted_df['res_dir_y_1_HL'] = weighted_df.groupby(['level', 'path', 'node_x_1', 'node_y_1'])['res_dir_y_1_HL'].transform('sum') / (.0001 + weighted_df['aggregated_weight'])
    # weighted_df['res_dir_x_1_LH'] = weighted_df.groupby(['level', 'path', 'node_x_1', 'node_y_1'])['res_dir_x_1_LH'].transform('sum') / (.0001 + weighted_df['aggregated_weight'])
    # weighted_df['res_dir_y_1_LH'] = weighted_df.groupby(['level', 'path', 'node_x_1', 'node_y_1'])['res_dir_y_1_LH'].transform('sum') / (.0001 + weighted_df['aggregated_weight'])
    """
    # jyoti

    weighted_df['res_dir_x_1_HL'] = weighted_df.groupby(['level', 'path'])['res_dir_x_1_HL'].transform('sum') / (weighted_df['aggregated_weight']+.00000001)
    weighted_df['res_dir_y_1_HL'] = weighted_df.groupby(['level', 'path'])['res_dir_y_1_HL'].transform('sum') / (weighted_df['aggregated_weight']+.00000001)
    weighted_df['res_dir_x_1_LH'] = weighted_df.groupby(['level', 'path'])['res_dir_x_1_LH'].transform('sum') / (weighted_df['aggregated_weight']+.00000001)
    weighted_df['res_dir_y_1_LH'] = weighted_df.groupby(['level', 'path'])['res_dir_y_1_LH'].transform('sum') / (weighted_df['aggregated_weight']+.00000001)

    """

    # weighted_df['resultant_pos'] = weighted_df['res_dir_x_1_HL'] + weighted_df['res_dir_y_1_HL']
    # weighted_df['resultant_neg'] = weighted_df['res_dir_x_1_LH'] + weighted_df['res_dir_y_1_LH']
    # weighted_df['mag_pos'] = np.sqrt(np.square(weighted_df['res_dir_x_1_pos']) + np.square(weighted_df['res_dir_y_1_pos']))
    # weighted_df['mag_neg'] = np.sqrt(np.square(weighted_df['res_dir_x_1_neg']) + np.square(weighted_df['res_dir_y_1_neg']))

    weighted_df['resultant'] = 0  # weighted_df['res_dir_x_1'] + weighted_df['res_dir_y_1']
    weighted_df['mag_HL'] = np.sqrt(np.square(weighted_df['res_dir_x_1_HL']) + np.square(weighted_df['res_dir_y_1_HL']))
    weighted_df['mag_LH'] = np.sqrt(np.square(weighted_df['res_dir_x_1_LH']) + np.square(weighted_df['res_dir_y_1_LH']))

    print("For creating final vector %s seconds" % (time.time() - final_vector))
    return weighted_df


def filterBasedOnGrid(depth, weighted_graph):
    """Grid"""
    quadtree = time.time()
    points = list(zip(weighted_graph['node_x'].tolist(), weighted_graph['node_y'].tolist()))
    qtree = QTree(depth, points)
    print ("filter2")
    qtree.subdivide()
    h = qtree.graph()
    print ("filter3")
    x_grid = np.unique(np.array(h[0])//1).tolist()
    y_grid = np.unique(np.array(h[1])//1).tolist()
    print ("filter4")
    v1 = np.isin(weighted_graph['node_x'], x_grid)
    v2 = np.isin(weighted_graph['node_y'], y_grid)
    v3 = v1 | v2
    weighted_graph = weighted_graph[v3]
    print("For creating quadtree %s seconds" % (time.time() - quadtree))
    return weighted_graph


def assignColor(data, column_name):
    color_1 = data[data[column_name] > data[column_name].median()]
    color_2 = data[data[column_name] <= data[column_name].median()]
    green_list = np.empty(len(color_1), dtype=object)
    green_list[:] = 'green'
    blue_list = np.empty(len(color_2), dtype=object)
    blue_list[:] = 'lime'
    color_1['color'] = green_list.tolist()
    color_2['color'] = blue_list.tolist()
    colored_df = pd.concat([color_1, color_2])
    colored_df = colored_df.sort_values(['level', 'path'])
    return colored_df

