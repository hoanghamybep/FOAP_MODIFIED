import numpy as np
import pandas as pd
import os
from extract_flow_features import extract_flow_features


def dirpath(lpath, index_table):
    lfilelist = []
    list = os.listdir(lpath)
    for f in list:
        file = os.path.join(lpath, f)
        if os.path.isdir(file):
            dirpath(file, lfilelist)
        else:
            if f.__contains__('.txt') is False:
                continue
            strs = f.split('.')
            strs_1 = strs[0].split('_')
            label = int(strs_1[0])
            sample_index = int(strs_1[1])
            if index_table.__contains__(sample_index) is False:
                continue
            lfilelist.append((file,label,sample_index))
    return lfilelist


def read_file(app_label, file_info, time_offset):
    file_name = file_info[0]
    app_label = str(app_label)
    activity_label = str(file_info[1])
    with open(file_name,'r') as file:
        records = file.readlines()
    flow_table = {}
    is_first = True
    stop_time = 0
    for line in records:
        strs = line.strip('\n').split(',')
        arrival_time = float(strs[0])

        if is_first:
            is_first = False
            start_time = arrival_time
        packet_size = int(strs[6])
        direction = int(strs[1])
        real_time = arrival_time - start_time + time_offset
        stop_time = real_time
        if direction == 1:
            end_points = (strs[2], strs[4], strs[3], strs[5])
        else:
            end_points = (strs[3], strs[5],strs[2], strs[4])
        if flow_table.__contains__(end_points) is False:
            function_label = activity_label+'#' + strs[3]+'#'+strs[5]
            flow_table[end_points] = [real_time, function_label, []]
        flow_table[end_points][2].append([real_time, direction, packet_size])

    items = []
    for end_points in flow_table.keys():
        time, function_label, flow = flow_table[end_points]
        if len(flow) <= 3:
            continue
        features = extract_flow_features(flow)
        item = [app_label, activity_label, function_label, time]
        item.extend(features)
        items.append(item)

    items = sorted(items, key=lambda x: x[3])

    return items, stop_time


def get_app_info():
    activity_num = 10
    app_num = 100
    app_table = {}
    for i in range(app_num):
        app_label = i
        if app_table.__contains__(app_label) is False:
            app_table[app_label] = []
        for j in range(activity_num):
            index = i*activity_num+j
            activity_label = index
            app_table[app_label].append(activity_label)
    return app_table


if __name__ == '__main__':
    data_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_data/web_sample/'
    output_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_data/metric_learning_sample/'

    app_table = get_app_info()
    sample_num = 5
    for app_label in app_table.keys():
        activity_list = app_table[app_label]
        for i in range(sample_num):
            time_offset = 0
            traffic = []
            for j in activity_list:
                file_name = data_path+str(j)+'_'+str(i)+'.txt'
                if os.path.exists(file_name) is False:
                    continue
                file_info = [file_name, j]
                print('Process '+file_name)
                items, time_offset = read_file(app_label, file_info, time_offset)
                traffic.extend(items)
            output_file_name = output_path+str(app_label)+'_'+str(i)+'.txt'
            traffic = np.array(traffic)
            df = pd.DataFrame(traffic)
            df.to_csv(output_file_name,header=False,index=False)

    print('Done!')