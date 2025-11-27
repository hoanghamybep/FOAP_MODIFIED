import numpy as np
import datetime


min_segment_interval = 10
min_segment_flow_num = 10
merge_threshold = 1


def get_split_point(xs):
    zs = xs[:,0]
    times = xs[:,1]
    if len(zs) <= 1:
        return -1, 0
    var_0 = np.var(zs)*len(zs)
    d_vars = []
    for i in range(1,len(zs)):
        zs_1 = zs[:i]
        zs_2 = zs[i:]
        times_1 = times[:i]
        times_2 = times[i:]
        if times_1[-1]-times_1[0] < min_segment_interval or times_2[-1]-times_2[0] < min_segment_interval:
            d_vars.append(-1)
            continue
        if len(zs_1) < min_segment_flow_num or len(zs_2) < min_segment_flow_num:
            d_vars.append(-1)
            continue

        var_1 = np.var(zs_1)*len(zs_1)+np.var(zs_2)*len(zs_2)
        d_var = var_0-var_1
        d_vars.append(d_var)
    index = np.argmax(np.array(d_vars))
    return index+1, d_vars[index]


def split_and_compute(xs, split_index):
    xs_1 = xs[:split_index,:]
    xs_2 = xs[split_index:,:]
    return xs_1, xs_2


def segment_trace(times, probability, my_merge_threshold=None):
    if my_merge_threshold != None:
        global merge_threshold
        merge_threshold = my_merge_threshold
    start_time = datetime.datetime.now()
    print('traffic_segmentation')
    xs = np.hstack([probability.reshape(-1, 1), times.reshape(-1, 1), np.array(list(range(len(times)))).reshape(-1, 1)])
    traces = [xs]
    split_info_list = []
    split_point, d_var = get_split_point(traces[0])
    split_info_list.append([split_point, d_var])

    while True:
        best_index = np.argmax(np.array(split_info_list)[:, 1])
        best_split_point, max_d_var = split_info_list[best_index]
        if max_d_var <= 0:
            break
        xs_1, xs_2 = split_and_compute(traces[best_index], best_split_point)
        del traces[best_index]
        traces.insert(best_index, xs_2)
        traces.insert(best_index, xs_1)
        del split_info_list[best_index]
        split_point_2, d_var_2 = get_split_point(xs_2)
        split_info_list.insert(best_index, [split_point_2, d_var_2])
        split_point_1, d_var_1 = get_split_point(xs_1)
        split_info_list.insert(best_index, [split_point_1, d_var_1])

    print('complete splitting, segment num: '+str(len(traces)))

    d_var_list = []
    for i in range(len(traces) - 1):
        d_var = get_variance_increase(traces[i], traces[i + 1])
        d_var_list.append(d_var)
    while len(traces)>1:
        merge_index = np.argmin(np.array(d_var_list))
        if d_var_list[merge_index] > merge_threshold:
            break
        traces[merge_index] = np.vstack([traces[merge_index], traces[merge_index+1]])
        del traces[merge_index+1]
        if merge_index != 0:
            pre_d_var = get_variance_increase(traces[merge_index-1], traces[merge_index])
            d_var_list[merge_index-1] = pre_d_var
        if merge_index != len(d_var_list) - 1:
            post_d_var = get_variance_increase(traces[merge_index], traces[merge_index+1])
            d_var_list[merge_index+1] = post_d_var
        del d_var_list[merge_index]

    print('complete merging, segment num: '+str(len(traces)))
    bound_list = []
    for i in range(len(traces)):
        trace = traces[i]
        bound_1 = int(trace[0, -1])
        bound_2 = int(trace[-1, -1])
        for j in range(trace.shape[0]):
            bound_list.append([bound_1, bound_2])
    return bound_list


def get_variance_increase(trace_1, trace_2):
    zs_1 = trace_1[:,0]
    zs_2 = trace_2[:,0]
    d_var = np.var(np.concatenate([zs_1, zs_2]))*(len(zs_1)+len(zs_2))-np.var(zs_1)*len(zs_1)-np.var(zs_2)*len(zs_2)
    return d_var


