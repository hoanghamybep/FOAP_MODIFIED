import numpy as np
import datetime
import collections
from extract_flow_features import extract_flow_features
from os.path import exists, getsize
import pickle
import pandas as pd


force_generate_sample = True
#force_generate_sample = False
burst_interval_threshold = 0.5
temporal_context_threshold = 0.5
min_flow_packet_num = 3
is_burst_method = False
old_date = datetime.datetime.strptime('1990-01-01', '%Y-%m-%d')
future_date = datetime.datetime.strptime('2990-01-01', '%Y-%m-%d')


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

    print(str(len(flow_table))+' flow records from pcap')
    return flow_table


def parse_log(path):
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    flow_method_table = {}
    current_flow_name = ''
    for i in range(len(lines)):
        if lines[i].__contains__('flow:'):
            line = lines[i].strip('\n')
            flow_name = line[5:]
            flow_method_table[flow_name] = []
            current_flow_name = flow_name
            continue
        elif lines[i].__contains__('socket:') is False:
            continue
        line = lines[i].strip('\n')
        line = line[7:]
        strs = line.split(',')
        if len(strs) != 4:
            continue
        time = strs[0]
        write_time = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')
        method = strs[-1]
        flow_method_table[current_flow_name].append([method, write_time])
    return flow_method_table


def label_flow(flow_table, flow_method_table):
    offset = 0.001
    interval = 1
    sample_table = {}
    for flow_name in flow_table.keys():

        start_time = flow_table[flow_name][0]
        packets = flow_table[flow_name][2][:]
        if flow_method_table.__contains__(flow_name) is False:
            # methods = [['other', old_date]]
            continue
        else:
            methods = flow_method_table[flow_name]
            if len(methods) == 0:
                methods = [['unknown', old_date]]
            else:
                time_offset = np.round((methods[0][1] - start_time).total_seconds() / 3600)
                # start_time = start_time + datetime.timedelta(hours=time_offset)
                for i in range(len(methods)):
                    methods[i][1] = methods[i][1] - datetime.timedelta(hours=time_offset)

        packet_index = len(packets)-1
        flow_record = []
        for i in range(len(methods)-1,-1,-1):
            method = methods[i][0]
            write_time = (methods[i][1]-start_time).total_seconds()
            record = [method, write_time, []]
            for j in range(packet_index,-1,-1):
                if packets[j][0] >= write_time-offset:
                    record[2].insert(0, packets[j])
                else:
                    packet_index = j
                    break
            flow_record.insert(0, record)

        labeled_packets = []
        for i in range(len(flow_record)):
            method = flow_record[i][0]
            packets = list(map(lambda x:[x[0],x[1], x[2], method], flow_record[i][2]))
            labeled_packets.extend(packets)

        sample_table[flow_name] = [start_time, labeled_packets]
    return sample_table


def extract_flow_sample(flow_name, flow):
    start_time, labeled_packets = flow
    if len(labeled_packets) < min_flow_packet_num:
        return None, None
    methods = [x[3] for x in labeled_packets]
    packets = [x[:3] for x in labeled_packets]
    packets = np.array(packets)
    method_set_1 = set(methods)
    method_set_2 = set()
    packet_time = packets[:,0]
    packet_interval = packet_time[1:]-packet_time[:-1]
    index = (np.argwhere(packet_interval>burst_interval_threshold).flatten()+1).tolist()
    index.insert(0, 0)
    index.append(len(packet_time))
    bursts = []
    burst_index = []
    for i in range(1,len(index)):
        index_1 = index[i-1]
        index_2 = index[i]
        burst_index.append([index_1, index_2])
        method = collections.Counter(methods[index_1:index_2]).most_common(1)[0][0]
        method_set_2.add(method)
        burst = packets[index_1:index_2,:]
        assert len(burst)>0
        feature = extract_flow_features(burst)
        burst_time = start_time+datetime.timedelta(seconds=burst[0, 0])
        spatial_context = []
        temporal_context = []
        bursts.append([burst_time, method, flow_name, feature, spatial_context, temporal_context])
    # flow_sample = [list(method_set), bursts, packets[:,1], methods, burst_index]
    if is_burst_method is True:
        flow_sample = [list(method_set_2), bursts]
    else:
        flow_sample = [list(method_set_1), bursts]
    return flow_sample


def get_year(flow_table):
    for flow_name in flow_table.keys():
        year = str(flow_table[flow_name][0].year)
        return year


def generate_sample_burst(input_pcap_path, input_log_path, app_name, index):
    # method_file_name = input_path + 'parse_result/' +app_name + '_flowToMethod_'+str(index)+'.txt'
    method_file_name = input_log_path + 'logcat_' + app_name + '_' + str(index) + '_EP.txt'
    traffic_file_name = input_pcap_path + 'traffic_' + app_name + '_'+str(index)+'.txt'

    if exists(method_file_name) is False or exists(traffic_file_name) is False:
        return None
    if getsize(method_file_name) == 0 or getsize(traffic_file_name) == 0:
        return None
    flow_table = parse_pcap(traffic_file_name)
    flow_method_table = parse_log(method_file_name)
    labeled_flow_table = label_flow(flow_table, flow_method_table)

    sample_start_time = future_date
    for flow_name in labeled_flow_table.keys():
        if (labeled_flow_table[flow_name][0]-old_date).total_seconds()<=(sample_start_time-old_date).total_seconds():
            sample_start_time = labeled_flow_table[flow_name][0]

    flow_sample_table = {}
    burst_list = []

    for flow_name in labeled_flow_table.keys():
        label, bursts = extract_flow_sample(flow_name, labeled_flow_table[flow_name])
        if label is None:
            continue
        flow_start_time = labeled_flow_table[flow_name][0]
        flow_duration = labeled_flow_table[flow_name][1][-1][0]
        flow_sample_table[flow_name] = [label, [], [(flow_start_time-sample_start_time).total_seconds(), (flow_start_time-sample_start_time).total_seconds()+flow_duration]]
        for burst in bursts:
            flow_sample_table[flow_name][1].append(len(burst_list))
            burst_list.append(burst)

    for flow_name in flow_sample_table.keys():
        burst_index = flow_sample_table[flow_name][1]
        for i in burst_index:
            burst_list[i][4] = list(filter(lambda x: x != i, burst_index))

    for i in range(len(burst_list)):
        burst_time_1 = burst_list[i][0]
        for j in range(len(burst_list)):
            if i == j:
                continue
            if j in burst_list[i][4]:
                continue
            burst_time_2 = burst_list[j][0]
            if abs((burst_time_2-burst_time_1).total_seconds())<temporal_context_threshold:
                burst_list[i][5].append(j)

    return [flow_sample_table, burst_list]



def generate_sample(task_info):
    app_name, input_traffic_path, input_log_path, output_path, max_sample_num = task_info
    sample_index = list(range(max_sample_num))
    for index in sample_index:
        output_file_name = output_path + r'sample_' + app_name + '_' + str(burst_interval_threshold) + '_' + str(
            index) + '.pkl'
        if exists(output_file_name) is True and force_generate_sample is False:
            print('skip ' + output_file_name)
            continue
        flow_sample = None
        try:
            flow_sample = generate_sample_burst(input_traffic_path, input_log_path, app_name, index)
        except Exception as exc:
            print(exc)
        if flow_sample is None:
            continue
        try:
            with open(output_file_name, 'wb') as file:
                pickle.dump(flow_sample, file)
                print('output ' + output_file_name)
        except Exception as exc:
            print(exc)


def run():
    max_sample_num = 50
    app_list = np.array(pd.read_csv(r'E:/Paper/FOAP/Modify/foap_project/FOAP/app_list.txt', header=None)).flatten().tolist()
    app_list = sorted(app_list)
    for app_name in app_list:
        task_info = [app_name, my_input_traffic_path, my_input_log_path, my_output_path, max_sample_num]
        generate_sample(task_info)


if __name__ == '__main__':

    my_input_traffic_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_data/traffic/'
    my_input_log_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_data/log/'
    my_output_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_data/method_sample/'
    run()
    print('Done!')