import numpy as np
import random
import torch
import pickle
from flow_embedding import FlowEmbedding
import itertools
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from os.path import exists
from os import walk
from sklearn.ensemble import RandomForestClassifier
from trace_segmentation import segment_trace
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score
import pandas as pd


flow_embedding_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_model/flow_embedding.model'
model_folder = r'E:/Paper/FOAP/Modify/foap_project/FOAP_model/'

# output_detail = True
output_detail = False


force_training = False
# force_training = True
is_test = True
# is_test = False

sample_weight = 0.1
positive_num = 1
ratio = 0.7
context_window_size = 10
clustering_threshold =0.5
extension_window = 10
negative_ratio = 10
negative_ratio_logistic = 1
noise_weight = 1
low_bound = 0.7
square_threshold_lower_bound = 0.1
square_threshold_upper_bound = 0.25
app_interval = 1
batch_num = 5
my_merge_threshold = 0.5
background_threshold = 0.2


def get_app_list(app_data_path):
    app_list = []
    for dir_path, folders, files in walk(app_data_path):
        for file_name in files:
            if file_name.__contains__('.pkl') is False:
                continue
            strs = file_name.split('.')
            if len(strs) == 2:
                app_name = strs[0]
            else:
                app_name = '.'.join(strs[:-1])
            app_list.append(app_name)
    app_list = list(set(app_list))
    app_list = sorted(app_list)
    return app_list


def get_sample_context(time_list, window_size):
    sample_context = []
    start_index = 0
    for i in range(len(time_list)):
        time = time_list[i]
        index1 = 0
        index2 = 0
        for j in range(len(time)):
            while time[index1] < time[j] - window_size:
                index1 += 1
            while index2 < len(time) - 1 and time[index2 + 1] < time[j] + window_size:
                index2 += 1
            context = [index + start_index for index in range(index1, index2 + 1)]
            sample_context.append(context)
        start_index += len(time)
    return sample_context


def flatten_data_list(data_list):
    new_data_list = []
    for i in range(len(data_list)):
        for j in range(len(data_list[i])):
            data = data_list[i][j]
            new_data_list.append(data)

    my_data = np.vstack(new_data_list)
    return my_data


def get_methods(data):
    methods = list(set(list(filter(lambda x:x!=-1,data[:,0].tolist()))))
    return sorted(methods)


def train(positive_training_data, noise_training_data, negative_training_data):
    positive_training_data[:,0] = 1
    noise_training_data[:,0] = 0
    negative_training_data[:,0] = -1
    negative_weight = float(positive_training_data.shape[0])/negative_training_data.shape[0]*negative_ratio
    print('positive sample num:'+str(positive_training_data.shape[0]))
    # negative_weight = 1
    weights = []
    weights.extend([1]*positive_training_data.shape[0])
    weights.extend([1] * noise_training_data.shape[0])
    weights.extend([negative_weight] * negative_training_data.shape[0])
    training_data = np.vstack([positive_training_data, noise_training_data, negative_training_data])
    labels = training_data[:,0].astype(int)
    features = training_data[:,1:]
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    model.fit(features, labels, sample_weight=weights)
    return model


def preprocess_negative_data_list(data_list):
    for i in range(len(data_list)):
        for j in range(len(data_list[i])):
            data_list[i][j] = data_list[i][j][:,1:]
    return data_list


def preprocess_data_list(data_list):
    for j in range(len(data_list)):
        data_list[j] = data_list[j][:, 1:]
    return data_list


def load_flow_embedding():
    checkpoint = torch.load(flow_embedding_path)
    flow_embedding.load_state_dict(checkpoint['model_state_dict'])
    flow_embedding.eval()


def data_clustering(data):
    labels = data[:,0]
    noise_index = np.argwhere(labels < 0).flatten().tolist()
    labels = np.ones(len(labels))
    labels[noise_index] = -1
    main_len = int(np.max(np.argwhere(labels<10000).flatten()))+1

    original_features = data[:,1:]
    features = torch.from_numpy(np.array(original_features, dtype='float32'))
    zs = flow_embedding(features)
    xs = zs.detach().numpy()
    new_labels = labels.tolist()

    index = np.argwhere(labels == 1).flatten().tolist()
    points = xs[index, :]
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=clustering_threshold,linkage='single').fit(points)
    cluster_labels = clustering.labels_
    for i in range(len(index)):
        new_labels[index[i]] = str(labels[index[i]]) + '#' + str(cluster_labels[i])

    label_table = {}
    max_main_label = 0
    noise_list = []
    main_method_list = []
    cluster_table = {}

    for i in range(len(new_labels)):
        label = new_labels[i]

        if label_table.__contains__(label) is False:
            label_table[label] = len(label_table)
            if label == -1:
                noise_list.append(label_table[label])
            else:
                main_method_list.append(label_table[label])

        cluster_index = label_table[label]
        if cluster_table.__contains__(cluster_index) is False:
            cluster_table[cluster_index] = 0
        cluster_table[cluster_index] += 1

        new_labels[i] = label_table[label]

    data = np.hstack([np.array(new_labels).reshape(-1,1), original_features])
    method_list = [main_method_list, noise_list]

    print('cluster num:'+str(len(label_table)))

    main_index = np.argwhere(np.array(new_labels)!=noise_list[0]).flatten().tolist()
    metric_features = xs[main_index,:]
    metric_labels = np.array(new_labels)[main_index]
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(metric_features, metric_labels)
    return data, method_list, neigh


def get_time_list(data_list):
    time_list = []
    offset = 0
    for i in range(len(data_list)):
        for j in range(len(data_list[i])):
            time = data_list[i][j][:, 0].reshape(-1,1) + offset
            time_list.append(time)
            offset = np.max(time) + np.random.exponential(app_interval,1)
            # offset = np.max(time) + app_interval
    return time_list


def compute_score_1(data, RF_model, neigh_model):
    labels = data[:,0]
    features = data[:,1:]

    # --- Thêm đoạn kiểm tra và làm sạch NaN ở đây ---
    features = features.astype(np.float32)
    if np.isnan(features).any():
        #print("Cảnh báo: Phát hiện NaN trong dữ liệu đặc trưng. Sẽ thay thế bằng 0.")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    # --- Kết thúc phần thêm vào ---

    feature_num = features.shape[1]
    global flow_embedding
    flow_embedding = FlowEmbedding(feature_num)
    load_flow_embedding()

    features_1 = torch.from_numpy(np.array(features, dtype='float32'))
    zs = flow_embedding(features_1)
    xs = zs.detach().numpy()

    # --- Thêm một bước kiểm tra an toàn nữa cho đầu ra của model PyTorch ---
    if np.isnan(xs).any():
        #print("Cảnh báo: Đầu ra của flow_embedding chứa NaN. Sẽ thay thế bằng 0.")
        xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)
    # --- Kết thúc phần thêm vào ---

    metric_features = xs
    clusters = neigh_model.predict(metric_features)
    probabilities = RF_model.predict_proba(features)
    results = []
    if probabilities.shape[1] == 2:
        probabilities = np.hstack([probabilities[:, 0].reshape((-1,1)), np.zeros((probabilities.shape[0] ,1)) ,probabilities[:, 1].reshape((-1,1))])

    for i in range(len(labels)):
        label = labels[i]
        index = clusters[i]
        score_1 = probabilities[i, 2]
        score_2 = probabilities[i, 1]
        result = [label, index, score_1, score_2]
        results.append(result)
    results = np.array(results)
    return results


def get_diversity(correlation_scores, correlation_clusters):
    correlation_table = {}
    for i in range(len(correlation_clusters)):
        cluster = correlation_clusters[i]
        if correlation_table.__contains__(cluster) is False:
            correlation_table[cluster] = 0
        if correlation_table[cluster]<correlation_scores[i]:
            correlation_table[cluster] = correlation_scores[i]
    my_scores = [correlation_table[x] for x in correlation_table.keys()]
    diversity = np.sum(np.array(my_scores))/len(correlation_scores)
    return diversity


def recognize_app(active_app, results, context_time_list, background_list, clf_1, logistic_threshold, square_threshold, times, RF_prob):
    labels = results[:,0]
    scores_1 = results[:, 2]
    scores_2 = results[:, 3]
    clusters = results[:,1]
    times = np.array(times)
    intervals = times[1:]-times[:-1]
    intervals = intervals.tolist()
    intervals.insert(0,0)

    backgrounds = list(set(map(lambda x:tuple(x), background_list)))
    active_segment_table = {}
    for background in backgrounds:
        index = np.argwhere((times >= times[background[0]]-extension_window) & (times <= times[background[1]]+extension_window)).flatten().tolist()
        active_segment_table[background] = index

    square_threshold = min(max(square_threshold, square_threshold_lower_bound), square_threshold_upper_bound)

    ps = []
    xs = results[:,1:4]
    square_table = {}
    squares = []
    for i in range(len(scores_1)):
        background = background_list[i]
        background = tuple(background)
        if square_table.__contains__(background):
            square_1 = square_table[background]
        else:
            x = xs[background[0]: background[1] + 1, :]
            square_1, _ = get_square(x)

            square_table[background] = square_1
            if square_1 < square_threshold:
                del active_segment_table[background]
        squares.append(square_1)

    for i in range(len(scores_1)):
        context_index = context_time_list[i]
        context = scores_1[context_index]
        feature_1 = np.array([scores_1[i], scores_1[i] - scores_2[i], np.mean(context)])
        p = clf_1.predict_proba(feature_1.reshape(1, -1))
        p = p[0][1]
        ps.append(p)

    ps = np.array(ps)

    active_index = []
    for background in active_segment_table.keys():
        active_index.extend(active_segment_table[background])
    active_index = sorted(list(set(active_index)))
    inert_index = sorted(list(set(list(range(len(labels))))-set(active_index)))
    ps[inert_index] = 0

    positive_index = np.argwhere(labels >= 0).flatten().tolist()
    negative_index = np.argwhere(labels < 0).flatten().tolist()
    mean_score_1 = np.mean(scores_1[positive_index])
    mean_score_2 = np.mean(scores_1[negative_index])
    # print('positive score:' + str(mean_score_1) + ', negative score:' + str(mean_score_2))
    ps = np.array(ps)
    ps = ps-(logistic_threshold-0.5)
    ps = np.round(ps)
    labels = np.ones(len(labels))
    labels[negative_index] = 0
    precision, recall, F1 = performance_evaluation(ps, labels)
    print('FOAP: '+active_app+', precision:' + str(precision) + ', recall:' + str(recall)+', F1-score:'+str(F1))
    predictions = ps
    return precision, recall, F1, predictions, labels


def performance_evaluation(predictions, labels):
    precision = precision_score(labels, predictions, zero_division=1)
    recall = recall_score(labels, predictions, zero_division=1)
    if precision+recall == 0:
        F1 = 0
    else:
        F1 = 2*precision*recall/(precision+recall)
    precision = round(precision, 4)
    recall = round(recall,4)
    return precision, recall, F1


def modify_negative_sample_label(data_list):
    for i in range(len(data_list)):
        index = np.argwhere(data_list[i][:,1] !=-2).flatten().tolist()
        data_list[i][index, 1] = -1
    return data_list


def get_flow_info(app_name, flow_method_list):
    info_list = []
    for i in range(len(flow_method_list)):
        infos = []
        for j in range(len(flow_method_list[i])):
            if flow_method_list[i][j] == 'other':
                infos.append('other')
            else:
                infos.append(app_name)
        info_list.append(infos)
    return info_list


def test_app_model(active_app, test_num=20, training_app_num=10):

    model_path = model_folder + r'new_logistic_models_' + str(active_app)+'_'+str(training_app_num) + '.model'
    multiple_model_path = model_folder + r'multiple_model_' + str(active_app) +'_'+str(training_app_num)+ '.model'
    with open(multiple_model_path, 'rb') as file:
        model, neigh_model, method_list, negative_app_index_list = pickle.load(file)
    with open(model_path, 'rb') as file:
        logistic_model, logistic_threshold, square_threshold = pickle.load(file)
    logistic_threshold = np.minimum(np.maximum(logistic_threshold, low_bound), 1)
    app_index_list = get_app_list(testing_app_data_path)
    app_index_list = list(filter(lambda x:x!=active_app, app_index_list))
    random.shuffle(app_index_list)
    testing_data_list = []
    testing_method_list = []
    testing_info_list = []
    app_file_name = testing_app_data_path + str(active_app) + '.pkl'
    with open(app_file_name, 'rb') as file:
        data_list = pickle.load(file)
    testing_data_list.extend(data_list[1])
    testing_method_list.extend(data_list[3])
    testing_info_list.extend(get_flow_info(active_app, data_list[3]))

    testing_app_index_list = list(set(app_index_list)-set(negative_app_index_list))
    random.shuffle(testing_app_index_list)
    testing_app_index_list = testing_app_index_list[:test_num]

    for app_index in testing_app_index_list:
        app_file_name = testing_app_data_path + str(app_index) + '.pkl'
        with open(app_file_name, 'rb') as file:
            data_list = pickle.load(file)
        negative_data_list = data_list[1]
        testing_method_list.extend(data_list[3])
        testing_info_list.extend(get_flow_info(app_index, data_list[3]))
        negative_data_list = modify_negative_sample_label(negative_data_list)
        testing_data_list.extend(negative_data_list)

    # randomize testing sample order
    testing_sample_index = list(range(len(testing_data_list)))
    random.shuffle(testing_sample_index)
    temp_testing_data_list = []
    temp_testing_method_list = []
    temp_testing_info_list = []
    for index in testing_sample_index:
        temp_testing_data_list.append(testing_data_list[index])
        temp_testing_method_list.append(testing_method_list[index])
        temp_testing_info_list.append(testing_info_list[index])
    testing_data_list = temp_testing_data_list
    testing_method_list = temp_testing_method_list
    testing_method_list = list(itertools.chain(*testing_method_list))
    testing_info_list = temp_testing_info_list
    testing_info_list = list(itertools.chain(*testing_info_list))

    times = get_time_list([testing_data_list])
    times = np.vstack(times).flatten()
    context_list = get_sample_context([times], context_window_size)
    testing_data_list = preprocess_negative_data_list([testing_data_list])

    testing_data = flatten_data_list(testing_data_list)
    results = compute_score_1(testing_data, model, neigh_model)
    RF_prob = results[:,2]

    background_list = segment_trace(times, RF_prob, my_merge_threshold=my_merge_threshold)
    precision_2, recall_2, F1_2, predictions, labels = recognize_app(active_app, results, context_list, background_list, logistic_model, logistic_threshold, square_threshold, times, RF_prob)
    if output_detail:
        output_recognition_detail(active_app, training_app_num, predictions, labels, testing_info_list)
    print('#'*100)
    RF_predictions = np.zeros(len(RF_prob))
    positive_index = np.argwhere(RF_prob>=0.5).flatten().tolist()
    RF_predictions[positive_index] = 1
    raw_labels = results[:,0]
    positive_index = np.argwhere(raw_labels >= 0).flatten().tolist()
    labels = np.zeros(len(raw_labels))
    labels[positive_index] = 1
    precision_1, recall_1, F1_1 = performance_evaluation(RF_predictions, labels)
    return precision_2, recall_2, F1_2


def output_recognition_detail(app_name, training_app_num, predictions, labels, info_list):
    predictions = predictions.tolist()
    labels = labels.tolist()
    for i in range(len(predictions)):
        if np.abs(predictions[i])==0:
            predictions[i] = np.abs(predictions[i])
    result = list(zip(labels, predictions, info_list))
    output_file_name = detail_folder+app_name + '_' + str(training_app_num)+'.txt'
    df = pd.DataFrame(result)
    df.to_csv(output_file_name, header=False, index=False)
    print('output ' + output_file_name)


def load_negative_app_training_data(negative_app_index_list):
    negative_data_list = []
    for app_index in negative_app_index_list:
        app_file_name = training_app_data_path + str(app_index) + '.pkl'
        with open(app_file_name, 'rb') as file:
            data_list = pickle.load(file)
        negative_data_list.append(data_list[0])
    negative_data_list = list(itertools.chain(*negative_data_list))
    negative_data_list = np.vstack(negative_data_list)
    negative_data_list = negative_data_list[:,1:]
    index = np.argwhere(negative_data_list[:,0] != -2).flatten().tolist()
    negative_data_list = negative_data_list[index,:].tolist()
    return negative_data_list


def load_negative_app_data(negative_app_index_list):
    training_data_list = []
    testing_data_list = []
    for app_index in negative_app_index_list:
        app_file_name = training_app_data_path + str(app_index) + '.pkl'
        with open(app_file_name, 'rb') as file:
            data_list = pickle.load(file)
        app_data = data_list[0]
        for i in range(len(app_data)):
            app_data[i][:,1] = -2
        random.shuffle(app_data)
        n = int(len(app_data) * ratio)
        training_app_data = app_data[:n]
        testing_app_data = app_data[n:]
        training_data_list.extend(training_app_data)
        testing_data_list.extend(testing_app_data)

    return training_data_list, testing_data_list


def get_square(x):
    score_bound_1 = 0.5
    score_bound_2 = 1
    x_table = {}
    for i in range(x.shape[0]):
        if x[i,2]>x[i,1] and x[i,2]>=background_threshold:
            continue
        cluster_index = x[i,0]
        score = x[i,1]
        if x_table.__contains__(cluster_index) is False:
            x_table[cluster_index] = []
        x_table[cluster_index].append(score)
    if len(x_table)==0:
        square = 0
        return square, len(x_table)
    xs = [[cluster_index, max(x_table[cluster_index])] for cluster_index in x_table.keys()]
    xs = np.array(xs)
    ps = np.arange(0, score_bound_2, 0.01)
    ys = []
    for p in ps:
        index = np.argwhere(xs[:,1] >= p)
        ys.append(len(index))

    ys = np.array(ys)
    ys = ys/ys[0]
    index = int(np.min(np.argwhere(ps>=score_bound_1)))
    square = np.mean(ys[index:])
    return square, len(x_table)


def get_square_threshold(squares):
    positive_index = np.argwhere(squares>0).flatten().tolist()
    squares = squares[positive_index]
    square_threshold = np.percentile(squares, 5)
    print('square threshold:'+str(square_threshold))
    return square_threshold


def train_identifier(results, context_time_list, background_list, model_path):
    labels = results[:, 0]
    scores_1 = results[:, 2]
    scores_2 = results[:, 3]
    positive_index = np.argwhere(labels >= 0).flatten().tolist()
    negative_index = np.argwhere(labels == -2).flatten().tolist()
    noise_index = np.argwhere(labels == -1).flatten().tolist()

    xs = results[:,1:4]
    square_list = []
    square_table = {}
    for i in positive_index:
        background = background_list[i]
        background = tuple(background)
        if square_table.__contains__(background):
            square = square_table[background]
        else:
            x = xs[background[0]: background[1] + 1, :]
            square, _ = get_square(x)
            square_table[background] = square
        square_list.append(square)
    my_square_threshold = get_square_threshold(np.array(square_list))

    features = []
    for i in range(len(labels)):
        context = scores_1[context_time_list[i]]
        feature_1 = [scores_1[i], scores_1[i] - scores_2[i], np.mean(context)]
        feature_1 = np.array(feature_1)
        features.append(feature_1)

    my_labels = np.zeros(len(labels))
    my_labels[positive_index] = 1
    weights = np.ones(len(labels))
    weights[negative_index] = float(len(positive_index))/len(negative_index)*negative_ratio_logistic
    weights[noise_index] = noise_weight

    clf_1 = LogisticRegression(random_state=0).fit(features, my_labels, weights)

    with open(model_path, 'wb') as file:
        pickle.dump([clf_1, 0.5, my_square_threshold], file)


def bootstrap(positive_sample_list, negative_sample_list, neigh_model):
    random.shuffle(positive_sample_list)
    random.shuffle(negative_sample_list)
    n = int(len(positive_sample_list) * ratio)
    raw_positive_training_data = positive_sample_list[:n]
    positive_testing_data = positive_sample_list[n:]

    n = int(len(negative_sample_list) * ratio)
    negative_training_data = negative_sample_list[:n]
    negative_training_data = np.vstack(negative_training_data)[:,1:]
    negative_testing_data = negative_sample_list[n:]

    raw_positive_training_data = np.vstack(raw_positive_training_data)[:,1:]
    positive_index = np.argwhere(raw_positive_training_data[:, 0] >= 0).flatten().tolist()
    noise_index = np.argwhere(raw_positive_training_data[:, 0] < 0).flatten().tolist()
    positive_training_data = raw_positive_training_data[positive_index, :]
    noise_training_data = raw_positive_training_data[noise_index, :]

    model = train(positive_training_data, noise_training_data, negative_training_data)

    testing_data_list = []
    testing_data_list.extend(positive_testing_data)
    testing_data_list.extend(negative_testing_data)
    random.shuffle(testing_data_list)
    times = get_time_list([testing_data_list])
    times = np.vstack(times).flatten()
    testing_data_list = preprocess_negative_data_list([testing_data_list])
    testing_data = flatten_data_list(testing_data_list)
    results = compute_score_1(testing_data, model, neigh_model)
    return results, times


def train_logistic(active_app, negative_app_index_list):
    model_path = model_folder+r'new_logistic_models_' + str(active_app) + '_' + str(len(negative_app_index_list))+'.model'

    negative_training_app_data, _ = load_negative_app_data(negative_app_index_list)

    app_file_name = training_app_data_path + str(active_app) + '.pkl'
    with open(app_file_name, 'rb') as file:
        data_list = pickle.load(file)

    training_data_list = data_list[0]
    positive_data = flatten_data_list(training_data_list)
    positive_data = positive_data[:,1:]

    feature_num = positive_data.shape[1] - 1
    global flow_embedding
    flow_embedding = FlowEmbedding(feature_num)
    load_flow_embedding()
    _, _, neigh_model = data_clustering(positive_data)

    batch_time_interval = 600
    time_offset = 0
    my_results = []
    my_times = []
    for _ in range(batch_num):
        results, times = bootstrap(training_data_list, negative_training_app_data, neigh_model)
        times = times+time_offset
        my_results.append(results)
        my_times.append(times)
        time_offset = np.max(times)+batch_time_interval

    my_results = np.vstack(my_results)
    my_times = np.concatenate(my_times)
    context_list = get_sample_context([my_times], context_window_size)
    RF_prob = my_results[:, 2]
    background_list = segment_trace(my_times, RF_prob, my_merge_threshold=my_merge_threshold)
    train_identifier(my_results, context_list, background_list, model_path)
    print('complete context-aware learning')


def train_app_model(active_app, app_num=5):
    app_file_name = training_app_data_path+str(active_app)+'.pkl'
    my_model_path = model_folder+r'multiple_model_'+str(active_app)+'_'+str(app_num)+'.model'

    app_index_list = get_app_list(training_app_data_path)
    app_index_list = list(filter(lambda x: x != active_app, app_index_list))
    random.shuffle(app_index_list)
    negative_app_index_list = app_index_list[:app_num]

    with open(app_file_name, 'rb') as file:
        data_list = pickle.load(file)
    training_data_list = data_list[0]
    training_data_list = preprocess_data_list(training_data_list)

    positive_data = list(itertools.chain(*training_data_list))
    random.shuffle(positive_data)
    positive_num = len(positive_data)

    feature_num = positive_data[0].shape[0]-1
    global flow_embedding
    flow_embedding = FlowEmbedding(feature_num)
    load_flow_embedding()

    raw_training_data = flatten_data_list([positive_data])

    raw_training_data, method_list, neigh_model = data_clustering(raw_training_data)
    noise_method = method_list[1][0]
    positive_index = np.argwhere(raw_training_data[:, 0] != noise_method).flatten().tolist()
    noise_index = np.argwhere(raw_training_data[:, 0] == noise_method).flatten().tolist()
    positive_training_data = raw_training_data[positive_index, :]
    noise_training_data = raw_training_data[noise_index, :]

    negative_app_data = load_negative_app_training_data(negative_app_index_list)
    negative_app_data = np.array(negative_app_data)
    model = train(positive_training_data, noise_training_data, negative_app_data)

    train_logistic(active_app, negative_app_index_list)

    with open(my_model_path, 'wb') as file:
        pickle.dump([model, neigh_model, method_list, negative_app_index_list], file)



def run(app_index, training_app_num=20, testing_app_num=20):
    app_list = get_app_list(training_app_data_path)
    if app_index >= len(app_list):
        print(str(app_index)+' is out of the range')
        return
    active_app = app_list[app_index]
    print('app:' + str(active_app) + '#' * 100)
    my_model_path = model_folder + r'multiple_model_' + str(active_app) + '_' + str(training_app_num) + '.model'
    if force_training is True or exists(my_model_path) is False:
        try:
            print('training stage:')
            train_app_model(active_app, training_app_num)
        except Exception as exc:
            print(exc)
    print('-' * 100)
    if is_test is False or exists(my_model_path) is False:
        return
    print('testing stage:')
    precision_2, recall_2, F1_2 = test_app_model(active_app, testing_app_num, training_app_num)


if __name__ == '__main__':
    detail_folder = r'temp/'

    training_app_data_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_data/flow_sample/'
    testing_app_data_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_data/flow_sample/'
    my_app_index = 0
    run(my_app_index, training_app_num=20, testing_app_num=20)
    print('Done!')

'''if __name__ == '__main__':
    detail_folder = r'temp/'
    training_app_data_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_data/flow_sample/'
    testing_app_data_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_data/flow_sample/'
    
    # 1. Lấy danh sách tất cả các ứng dụng có sẵn
    app_list = get_app_list(training_app_data_path)
    print(f"Tìm thấy {len(app_list)} ứng dụng để xử lý.")
    
    # 2. Sử dụng vòng lặp for để chạy qua từng ứng dụng
    for i in range(len(app_list)):
        # Lấy tên app hiện tại để in ra màn hình cho dễ theo dõi
        current_app_name = app_list[i] 
        print(f"\n{'='*50}\nBắt đầu xử lý ứng dụng {i+1}/{len(app_list)}: {current_app_name}\n{'='*50}")
        
        # Gọi hàm run với chỉ số (index) của ứng dụng hiện tại
        run(i, training_app_num=20, testing_app_num=20)
        
    print("\nĐã xử lý xong tất cả các ứng dụng!")
    print('Done!')'''
