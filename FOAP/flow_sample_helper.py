import numpy as np
import pandas as pd
import pickle
import random
from os.path import exists

# force_generating = False
force_generating = True
ratio = 0.8


def load_data(raw_data_path, active_app, sample_index):
    data_path = raw_data_path
    data_list = []
    for index in sample_index:
        file_name = data_path+str(active_app)+'_'+str(index)+'.txt'
        if exists(file_name) is False:
            continue
        tag = str(active_app)+'_'+str(index)
        X = np.array(pd.read_csv(file_name, header=None))
        features = X[:,2:]
        meta_infos = X[:,:2]                 
        data_list.append([meta_infos, features, tag])
    random.shuffle(data_list)
    return data_list


def update_active_label_table(meta_infos, active_label_table):
    labels = meta_infos[:,2]
    labels = labels.flatten().tolist()
    for label in labels:
        if label == '-1':
            continue
        if label == -1:
            continue
        if active_label_table.__contains__(label) is False:
            active_label_table[label] = len(active_label_table)
    return active_label_table


def prepare_output(data_list):
    new_data_list = []
    method_label_list = []
    for j in range(len(data_list)):
        meta_infos = data_list[j][0]
        n = meta_infos.shape[0]
        time = meta_infos[:, 0]
        meta_data = []
        multilabel_list = []
        for k in range(n):
            if meta_infos[k, 1] != 'other':
                label = 1
            else:
                label = -1
            meta_data.append([time[k], label])
            multilabel_list.append(meta_infos[k, 1])

        meta_data = np.array(meta_data)
        features = data_list[j][1]
        my_result = np.hstack([meta_data, features])
        new_data_list.append(my_result)
        method_label_list.append(multilabel_list)
    return new_data_list, method_label_list


def get_data_tag(data_list):
    tag_list = []
    for i in range(len(data_list)):
        tag_list.append(data_list[i][2])
    return tag_list


def preprocess_data(input_path, output_path, app_name, max_sample_num=50):
    active_app = app_name
    output_file_name = output_path + str(active_app) + '.pkl'
    if exists(output_file_name) is True and force_generating is False:
        print('skip ' + app_name)
        return
    sample_index = list(range(max_sample_num))
    my_data_list = load_data(input_path, active_app, sample_index)
    n_0 = int(np.ceil(ratio * len(my_data_list)))
    training_data_list = my_data_list[:n_0]
    testing_data_list = my_data_list[n_0:]

    training_data, training_method_list = prepare_output(training_data_list)
    testing_data, testing_method_list = prepare_output(testing_data_list)
          
    training_tags = get_data_tag(training_data_list)
    testing_tags = get_data_tag(testing_data_list)

    with open(output_file_name, 'wb') as file:
        pickle.dump([training_data, testing_data, training_method_list, testing_method_list, training_tags, testing_tags], file)
    print('output '+output_file_name)
