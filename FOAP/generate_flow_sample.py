import numpy as np
import re
from os import walk
import datetime
import collections
from extract_flow_features import extract_flow_features
from os.path import exists, getsize
import pandas as pd
from flow_sample_helper import preprocess_data


force_generate_sample = False
# force_generate_sample = True
min_flow_packet_num = 3


def get_local_ip(path):
    ip_list = []
    with open(path, 'r') as file:
        packets = file.readlines()

    for i in range(len(packets)):
        packet = packets[i].strip()
        strs = packet.split(',')
        timestamp, src_ip, sport, dst_ip, dport, packet_size = strs
        ip_list.append(src_ip)
        ip_list.append(dst_ip)

    counter = collections.Counter(ip_list)
    if len(counter) == 0:
        return -1
    local_ip = counter.most_common(1)[0][0]
    return local_ip


def parse_pcap(path):
    local_ip = get_local_ip(path)
    flow_table = {}
    is_first_packet = True
    if local_ip == -1:
        print(str(len(flow_table)) + ' flow records from pcap')
        return flow_table

    with open(path, 'r') as file:
        packets = file.readlines()

    for i in range(len(packets)):
        packet = packets[i].strip()
        strs = packet.split(',')
        timestamp, src_ip, sport, dst_ip, dport, packet_size = strs
        arrival_time = datetime.datetime.fromtimestamp(float(timestamp))
        if is_first_packet is True:
            is_first_packet = False
            start_time = arrival_time
        sport = int(sport)
        dport = int(dport)
        length = int(packet_size)
        if src_ip == local_ip:
            remote_ip = dst_ip
            remote_endpoint = remote_ip + '(' + str(dport) + ')'
            local_endpoint = src_ip + '(' + str(sport) + ')'
            flow_name = local_endpoint + '#' + remote_endpoint
            direction = 1
        else:
            remote_ip = src_ip
            remote_endpoint = remote_ip + '(' + str(sport) + ')'
            local_endpoint = dst_ip + '(' + str(dport) + ')'
            flow_name = local_endpoint + '#' + remote_endpoint
            direction = -1

        if flow_table.__contains__(flow_name) is False:
            flow_table[flow_name] = [arrival_time, remote_endpoint, []]

        time = (arrival_time - flow_table[flow_name][0]).total_seconds()
        flow_table[flow_name][2].append([time, direction, length])

    new_flow_table = {}
    for flow_name in flow_table.keys():
        flow_time = (flow_table[flow_name][0] - start_time).total_seconds()
        packets = flow_table[flow_name][2]
        if len(packets) < min_flow_packet_num:
            continue
        flow_features = extract_flow_features(packets)
        new_flow_table[flow_name] = [flow_time, 'other', flow_features]
    print(str(len(new_flow_table)) + ' flow records from pcap')
    return new_flow_table


def parse_log(path):
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    flow_method_table = {}
    current_flow_name = ''
    for i in range(len(lines)):
        if lines[i].__contains__('flow:'):
            line = lines[i].strip('\n')
            flow_name = line[5:]
            flow_method_table[flow_name] = set()
            current_flow_name = flow_name
            continue
        elif lines[i].__contains__('socket:') is False:
            continue
        line = lines[i].strip('\n')
        line = line[7:]
        strs = line.split(',')
        if len(strs) != 4:
            continue
        method = strs[-1]
        flow_method_table[current_flow_name].add(method)
    new_flow_method_table = {}
    for flow_name in flow_method_table.keys():
        method_list = list(flow_method_table[flow_name])
        if len(method_list) == 0:
            new_flow_method_table[flow_name] = 'unknown'
        else:
            new_flow_method_table[flow_name] = '#'.join(method_list)
    return new_flow_method_table


def generate_sample_flow(input_traffic_path, input_log_path, app_name, index):
    method_file_name = input_log_path + 'logcat_' + app_name + '_' + str(index) + '_EP.txt'
    traffic_file_name = input_traffic_path + 'traffic_' + app_name + '_'+str(index)+'.txt'

    if exists(method_file_name) is False or exists(traffic_file_name) is False:
        return None
    if getsize(method_file_name) == 0 or getsize(traffic_file_name) == 0:
        return None
    flow_method_table = parse_log(method_file_name)
    flow_table = parse_pcap(traffic_file_name)

    for flow_name in flow_table.keys():
        if flow_method_table.__contains__(flow_name) is True:
            flow_table[flow_name][1] = flow_method_table[flow_name]
    flow_sample_list = []
    for flow_name in flow_table.keys():
        flow_sample = [flow_table[flow_name][0], flow_table[flow_name][1]]
        flow_sample.extend(flow_table[flow_name][2])
        flow_sample_list.append(flow_sample)
    flow_sample_list = sorted(flow_sample_list, key=lambda x:x[0])
    return flow_sample_list


def list_app(path):
    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
        break
    filelist = []
    for file in f:
        if file.__contains__('_EP'):
            reg = re.compile(r'logcat_(\S+)_\d+_EP')
            strs = re.findall(reg, file)
            if len(strs) != 1:
                continue
            filelist.append(strs[0])
    filelist = list(set(filelist))
    return filelist


def generate_sample(task_info):
    app_name, input_traffic_path, input_log_path, sample_temp_path, output_path, max_sample_num = task_info
    sample_index = list(range(max_sample_num))
    for index in sample_index:
        output_file_name = sample_temp_path + app_name + '_' + str(index) + '.txt'
        if force_generate_sample is False and exists(output_file_name) is True:
            print('skip ' + output_file_name)
            continue
        try:
            flow_sample_list = generate_sample_flow(input_traffic_path, input_log_path, app_name, index)
        except Exception as exc:
            print(exc)
            continue

        if flow_sample_list is None:
            continue
        if len(flow_sample_list) == 0:
            continue
        df = pd.DataFrame(flow_sample_list)
        df.to_csv(output_file_name, header=False, index=False)
        print('output ' + output_file_name)
    preprocess_data(sample_temp_path, output_path, app_name, max_sample_num)


def run():
    max_sample_num = 50
    app_list = np.array(pd.read_csv(r'E:/Paper/FOAP/Modify/foap_project/FOAP/app_list.txt', header=None)).flatten().tolist()
    app_list = sorted(app_list)
    for app_name in app_list:
        task_info = [app_name, my_input_traffic_path, my_input_log_path, my_sample_temp_path, my_output_path, max_sample_num]
        generate_sample(task_info)


if __name__ == '__main__':

    my_input_traffic_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_data/traffic/'
    my_input_log_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_data/log/'
    my_sample_temp_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_data/temp/'
    my_output_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_data/flow_sample/'
    run()
    print('Done!')