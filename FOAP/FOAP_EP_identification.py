
import numpy as np
import pickle
from os.path import exists
from os import walk
from sklearn.ensemble import RandomForestClassifier
import random
import copy
import torch
from torch import nn
import collections
import datetime
import torch
import torch.nn
import itertools
from tqdm import tqdm


weight_ratio = 1.5
weight_0 = 2.852352
weight_1 = 0.5509616
rho = 0.1
ant_num = 100
iteration_num = 10
p_threshold = 0.9
min_leaf_samples = 5
gibbs_sample_n = 500
context_label_num = 40000000000
max_pair_num = 50000
weight_iteration_num = 10000
recognize_threshold = 0.0
max_compound_label_num = 100
batch_num = 2
compound_tree_num = 50

self_weight = 1
spatial_weight = 1
temporal_weight = 5
max_sample_num = 50

is_all_cpu = True
max_spatial_context = 50
max_temporal_context = 10
max_training_compound_sample = 200000
max_batch_test_sample_num = 100000


class weight_selection(nn.Module):
    def __init__(self):
        super(weight_selection,self).__init__()
        self.w_0 = nn.Parameter(torch.rand(1, dtype=torch.float, requires_grad=True))
        self.w_1 = nn.Parameter(torch.rand(1, dtype=torch.float, requires_grad=True))
        self.w_2 = nn.Parameter(torch.rand(1, dtype=torch.float, requires_grad=True))

    def forward(self, p_0, p_1, p_2):
        z = p_0*self.w_0 + p_1*self.w_1+p_2*self.w_2
        return z


def load_data(sample_file_list):
    sample_list = []
    for sample_file_name in sample_file_list:
        if exists(sample_file_name) is False:
            continue
        with open(sample_file_name, 'rb') as file:
            try:
                flow_sample = pickle.load(file)
            except:
                print('error in loading '+sample_file_name)
                continue
            if len(flow_sample[0])==0:
                continue
            if flow_sample is None:
                continue
            sample_list.append(flow_sample)
    return sample_list


def get_compound_label(label_1, label_2):
    if hash(label_1)<hash(label_2):
        return tuple([label_1, label_2])
    else:
        return tuple([label_2, label_1])


def get_training_sample(data_list):
    unary_samples = []
    compound_samples = []
    spatial_transition_table = {}
    temporal_transition_table = {}

    for _, burst_list in data_list:
        for i in range(len(burst_list)):
            _, method, _, feature, spatial_context, temporal_context = burst_list[i]
            random.shuffle(spatial_context)
            spatial_context = spatial_context[:max_spatial_context]
            random.shuffle(temporal_context)
            temporal_context = temporal_context[:max_temporal_context]

            unary_samples.append([method, feature])
            if spatial_transition_table.__contains__(method) is False:
                spatial_transition_table[method] = []
            if temporal_transition_table.__contains__(method) is False:
                temporal_transition_table[method] = []
            for j in spatial_context:
                _, method_1, _, feature_1, _, _ = burst_list[j]
                compound_label = (method, method_1)
                compound_feature = []
                compound_feature.extend(feature)
                compound_feature.extend(feature_1)
                compound_samples.append([compound_label, compound_feature])
                spatial_transition_table[method].append(method_1)
            for j in temporal_context:
                _, method_1, _, feature_1, _, _ = burst_list[j]
                compound_label = (method, method_1)
                compound_feature = []
                compound_feature.extend(feature)
                compound_feature.extend(feature_1)
                compound_samples.append([compound_label, compound_feature])
                temporal_transition_table[method].append(method_1)
    spatial_transition_p = {}
    for method in spatial_transition_table.keys():
        transition = collections.Counter(spatial_transition_table[method]).most_common()
        num = np.sum(np.array([x[1] for x in transition]))
        method_frequency = list(map(lambda x:[x[0],x[1]/num], transition))
        for method_1, p in method_frequency:
            compound_label = (method, method_1)
            spatial_transition_p[compound_label] = p

    temporal_transition_p = {}
    for method in temporal_transition_table.keys():
        transition = collections.Counter(temporal_transition_table[method]).most_common()
        num = np.sum(np.array([x[1] for x in transition]))
        method_frequency = list(map(lambda x: [x[0], x[1] / num], transition))
        for method_1, p in method_frequency:
            compound_label = (method, method_1)
            temporal_transition_p[compound_label] = p

    compound_label_p = {}
    for compound_label in spatial_transition_p.keys():
        if compound_label_p.__contains__(compound_label) is False:
            compound_label_p[compound_label] = spatial_transition_p[compound_label]
        else:
            compound_label_p[compound_label] += spatial_transition_p[compound_label]

    for compound_label in temporal_transition_p.keys():
        if compound_label_p.__contains__(compound_label) is False:
            compound_label_p[compound_label] = temporal_transition_p[compound_label]
        else:
            compound_label_p[compound_label] += temporal_transition_p[compound_label]

    compound_label_list = [[compound_label, compound_label_p[compound_label]] for compound_label in compound_label_p.keys()]
    compound_label_list = sorted(compound_label_list, key=lambda x:x[1], reverse=True)
    if len(compound_label_list)>max_compound_label_num:
        compound_label_list = compound_label_list[:max_compound_label_num]
    compound_label_set = set([x[0] for x in compound_label_list])
    random.shuffle(compound_samples)
    compound_samples = compound_samples[:max_training_compound_sample]
    return unary_samples, compound_samples, spatial_transition_p, temporal_transition_p, compound_label_set


def get_testing_sample(data_list):
    sample_list = []
    for flow_table, burst_list in data_list:
        burst_samples = []
        for i in range(len(burst_list)):
            _, method, _, feature, spatial_context, temporal_context = burst_list[i]
            random.shuffle(spatial_context)
            spatial_context = spatial_context[:max_spatial_context]
            random.shuffle(temporal_context)
            temporal_context = temporal_context[:max_temporal_context]
            burst_samples.append([method, feature, spatial_context, temporal_context])
        sample_list.append([flow_table, burst_samples])
    return sample_list


def train(active_app, burst_interval):
    setup_file_name = model_folder + r'setup_' + active_app + '_' + str(burst_interval) + '.pkl'
    with open(setup_file_name, 'rb') as file:
        training_sample_file_list = pickle.load(file)

    model_file_name = model_folder + r'crf_model_' + str(active_app) + '_' + str(burst_interval)+'.model'
    if force_training is False and exists(model_file_name) is True:
        return 0
    training_data_list = load_data(training_sample_file_list)
    if len(training_data_list) == 0:
        return -1
    unary_samples, compound_samples, spatial_transition_p, temporal_transition_p, compound_label_set = get_training_sample(training_data_list)
    if len(compound_samples) == 0:
        return -1
    compound_labels = list(compound_label_set)
    compound_label_table = {tuple(compound_labels[i]):i for i in range(len(compound_labels))}
    compound_samples = list(filter(lambda x:compound_label_set.__contains__(x[0]), compound_samples))

    labels = [x[0] for x in unary_samples]
    features = [x[1] for x in unary_samples]
    if is_all_cpu is True:
        unary_classifier = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    else:
        unary_classifier = RandomForestClassifier(n_estimators=200)


    time_0 = datetime.datetime.now()
    unary_classifier.fit(features, labels)
    time_1 = datetime.datetime.now()
    print('training unary classifier:' + str((time_1 - time_0).total_seconds()))

    labels = [compound_label_table[x[0]] for x in compound_samples]
    features = [x[1] for x in compound_samples]
    if is_all_cpu is True:
        compound_classifier = RandomForestClassifier(n_estimators=compound_tree_num, n_jobs=-1)
    else:
        compound_classifier = RandomForestClassifier(n_estimators=compound_tree_num)
    compound_classifier.fit(features, labels)
    time_2 = datetime.datetime.now()
    print('training compound classifier:' + str((time_2 - time_1).total_seconds()))

    # modify compound_label_table
    class_table = {compound_classifier.classes_[i]: i for i in range(len(compound_classifier.classes_))}
    bad_compound_labels = []
    for key in compound_label_table.keys():
        if class_table.__contains__(compound_label_table[key]) is False:
            bad_compound_labels.append(key)
            continue
        compound_label_table[key] = class_table[compound_label_table[key]]
    for bad_compound_label in bad_compound_labels:
        del compound_label_table[bad_compound_label]
    with open(model_file_name, 'wb') as file:
        pickle.dump(
            [unary_classifier, compound_classifier, spatial_transition_p, temporal_transition_p, compound_label_table], file)
        print('save '+model_file_name)
    return 0


def train_weight(active_app, burst_interval):
    setup_file_name = model_folder + r'setup_' + active_app + '_' + str(burst_interval) + '.pkl'
    with open(setup_file_name, 'rb') as file:
        training_sample_file_list = pickle.load(file)

    weight_file_name = model_folder + r'weight_' + str(active_app) + '_' + str(burst_interval)+'.model'
    if force_training is False and exists(weight_file_name) is True:
        return
    model_file_name = model_folder + r'crf_model_' + str(active_app) + '_' + str(burst_interval)+'.model'
    with open(model_file_name, 'rb') as file:
        global_unary_classifier, _, _, _, _ = pickle.load(file)
        print('load '+model_file_name)

    random.shuffle(training_sample_file_list)
    batch_size = np.ceil(len(training_sample_file_list)/batch_num)

    batch_list = []
    for i in range(len(training_sample_file_list)):
        index = int(i/batch_size)
        if len(batch_list) <= index:
            batch_list.append([])
        batch_list[index].append(training_sample_file_list[i])

    potentials = []
    labels = []
    for i in range(batch_num):
        print('process batch '+str(i))
        temp_batch_list = copy.deepcopy(batch_list)
        batch_testing_sample_file_list = temp_batch_list[i]
        del temp_batch_list[i]
        batch_training_sample_file_list = list(itertools.chain(*temp_batch_list))

        unary_classifier, compound_classifier, spatial_transition_p, temporal_transition_p, compound_label_table = batch_train(batch_training_sample_file_list)

        potential_list, label_index_list = batch_test(batch_testing_sample_file_list, unary_classifier,
                                                      compound_classifier, spatial_transition_p,
                                                      temporal_transition_p, compound_label_table,
                                                      global_unary_classifier.classes_)
        potentials.extend(potential_list)
        labels.extend(label_index_list)
    ps_0 = np.vstack([x[0].reshape(1, -1) for x in potentials])
    ps_1 = np.vstack([x[1].reshape(1, -1) for x in potentials])
    ps_2 = np.vstack([x[2].reshape(1, -1) for x in potentials])
    labels = np.array(labels)
    ps_0 = torch.from_numpy(ps_0).float()
    ps_1 = torch.from_numpy(ps_1).float()
    ps_2 = torch.from_numpy(ps_2).float()
    ys = torch.from_numpy(labels).long()

    weight_chooser = weight_selection()
    learning_rate = 0.1
    optimizer = torch.optim.Adam(weight_chooser.parameters(), lr=learning_rate)
    loss_f = torch.nn.CrossEntropyLoss()

    for t in range(weight_iteration_num):
        inputs = weight_chooser(ps_0, ps_1, ps_2)
        targets = ys
        loss = loss_f(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = loss.detach().numpy()
        my_weight_0 = weight_chooser.w_0.detach().numpy()[0]
        my_weight_1 = weight_chooser.w_1.detach().numpy()[0]
        my_weight_2 = weight_chooser.w_2.detach().numpy()[0]
        if t % 100 == 0 and show_info:
            print(str(t)+':loss:'+str(total_loss)+', weight 0:'+str(my_weight_0)+', weight 1:'+str(my_weight_1) + ', weight 2:' + str(my_weight_2))
    if my_weight_0 < 0:
        my_weight_0 = 1
    if my_weight_1 < 0:
        my_weight_1 = 1
    if my_weight_2 < 0:
        my_weight_2 = 1
    weights = [my_weight_0, my_weight_1, my_weight_2]

    with open(weight_file_name, 'wb') as file:
        pickle.dump(weights, file)


def batch_train(training_sample_file_list):
    # print('training sample:')
    # print(training_sample_file_list)
    training_data_list = load_data(training_sample_file_list)
    unary_samples, compound_samples, spatial_transition_p, temporal_transition_p, compound_label_set = get_training_sample(
        training_data_list)

    compound_labels = list(compound_label_set)
    compound_label_table = {tuple(compound_labels[i]): i for i in range(len(compound_labels))}
    compound_samples = list(filter(lambda x: compound_label_set.__contains__(x[0]), compound_samples))

    labels = [x[0] for x in unary_samples]
    features = [x[1] for x in unary_samples]
    if is_all_cpu is True:
        unary_classifier = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    else:
        unary_classifier = RandomForestClassifier(n_estimators=200)

    unary_classifier.fit(features, labels)

    labels = [compound_label_table[x[0]] for x in compound_samples]
    features = [x[1] for x in compound_samples]
    if is_all_cpu is True:
        compound_classifier = RandomForestClassifier(n_estimators=compound_tree_num, n_jobs=-1)
    else:
        compound_classifier = RandomForestClassifier(n_estimators=compound_tree_num)
    compound_classifier.fit(features, labels)

    class_table = {compound_classifier.classes_[i]: i for i in range(len(compound_classifier.classes_))}
    bad_compound_labels = []
    for key in compound_label_table.keys():
        if class_table.__contains__(compound_label_table[key]) is False:
            bad_compound_labels.append(key)
            continue
        compound_label_table[key] = class_table[compound_label_table[key]]
    for bad_compound_label in bad_compound_labels:
        del compound_label_table[bad_compound_label]

    return unary_classifier, compound_classifier, spatial_transition_p, temporal_transition_p, compound_label_table


def batch_test(batch_testing_sample_file_list, unary_classifier, compound_classifier, spatial_transition_p, temporal_transition_p, compound_label_table, global_method_class):
    print('batch test')
    testing_data_list = load_data(batch_testing_sample_file_list)
    sample_list = get_testing_sample(testing_data_list)

    method_table = {global_method_class[i]:i for i in range(len(global_method_class))}
    method_class = unary_classifier.classes_

    label_index_list = []
    potential_list = []
    for flow_table, burst_samples in sample_list:
        labels = [x[0] for x in burst_samples]
        features = [np.array(x[1]) for x in burst_samples]
        spatial_context = [x[2] for x in burst_samples]
        temporal_context = [x[3] for x in burst_samples]

        if len(features) == 0:
            continue
        probability = np.zeros((len(labels), len(global_method_class)))
        batch_probability = unary_classifier.predict_proba(features)
        for i in range(len(method_class)):
            if method_table.__contains__(method_class[i]) is False:
                print('error')
                continue
            method_index = method_table[method_class[i]]
            probability[:,method_index] = batch_probability[:,i]

        spatial_qs_list = []
        for i in range(len(spatial_context)):
            context_index = spatial_context[i]
            if len(context_index) > 0:
                context_features = []
                feature_1 = features[i]
                for j in context_index:
                    feature_2 = features[j]
                    feature = np.concatenate([feature_1, feature_2])
                    context_features.append(feature)
                qs = compound_classifier.predict_proba(context_features)
                spatial_qs_list.append(qs)
            else:
                spatial_qs_list.append([])

        temporal_qs_list = []
        for i in range(len(temporal_context)):
            context_index = temporal_context[i]
            if len(context_index) > 0:
                context_features = []
                feature_1 = features[i]
                for j in context_index:
                    feature_2 = features[j]
                    feature = np.concatenate([feature_1, feature_2])
                    context_features.append(feature)
                qs = compound_classifier.predict_proba(context_features)
                temporal_qs_list.append(qs)
            else:
                temporal_qs_list.append([])

        for i in range(len(spatial_context)):
            p_0 = probability[i, :]
            p_list = [p_0.reshape(1, -1)]
            context_index = spatial_context[i]
            if len(context_index) > 0:
                p_1 = np.zeros((len(context_index), len(p_0)))
                qs = spatial_qs_list[i]
                for k in range(len(context_index)):
                    for j in range(len(p_0)):
                        compound_label = (global_method_class[j], labels[context_index[k]])
                        if compound_label_table.__contains__(compound_label):
                            index = compound_label_table[compound_label]
                            q = qs[k, index]
                        else:
                            if method_table.__contains__(labels[context_index[k]]):
                                q = p_0[j] * probability[context_index[k], method_table[labels[context_index[k]]]]
                            else:
                                print('error')
                                q = 0
                        weight = 0
                        if spatial_transition_p.__contains__(compound_label):
                            weight = spatial_transition_p[compound_label]
                        p_1[k, j] = q * weight
                p_list.append(np.sum(p_1,axis=0))
            else:
                p_list.append(np.zeros((1, len(p_0))))
            context_index = temporal_context[i]
            if len(context_index) > 0:
                p_2 = np.zeros((len(context_index), len(p_0)))
                qs = temporal_qs_list[i]
                for k in range(len(context_index)):
                    for j in range(len(p_0)):
                        compound_label = (global_method_class[j], labels[context_index[k]])
                        if compound_label_table.__contains__(compound_label):
                            index = compound_label_table[compound_label]
                            q = qs[k, index]
                        else:
                            if method_table.__contains__(labels[context_index[k]]):
                                q = p_0[j] * probability[context_index[k], method_table[labels[context_index[k]]]]
                            else:
                                print('error')
                                q = 0
                        weight = 0
                        if temporal_transition_p.__contains__(compound_label):
                            weight = temporal_transition_p[compound_label]
                        p_2[k, j] = q * weight
                p_list.append(np.sum(p_2,axis=0))
            else:
                p_list.append(np.zeros((1, len(p_0))))

            if method_table.__contains__(labels[i]) is True:
                potential_list.append(p_list)
                label_index_list.append(method_table[labels[i]])
            else:
                print('error')
    return potential_list, label_index_list


def crf_greedy(burst_samples, unary_classifier, compound_classifier, spatial_transition_p, temporal_transition_p, compound_label_table):
    features = [np.array(x[1]) for x in burst_samples]
    spatial_context = [x[2] for x in burst_samples]
    temporal_context = [x[3] for x in burst_samples]
    method_class = unary_classifier.classes_

    probability = unary_classifier.predict_proba(features)

    predictions_1 = np.argmax(probability, axis=1).flatten()

    spatial_qs_list = []
    for i in range(len(spatial_context)):
        context_index = spatial_context[i]
        if len(context_index) > 0:
            context_features = []
            feature_1 = features[i]
            for j in context_index:
                feature_2 = features[j]
                feature = np.concatenate([feature_1, feature_2])
                context_features.append(feature)
            qs = compound_classifier.predict_proba(context_features)
            spatial_qs_list.append(qs)
        else:
            spatial_qs_list.append([])

    temporal_qs_list = []
    for i in range(len(temporal_context)):
        context_index = temporal_context[i]
        if len(context_index) > 0:
            context_features = []
            feature_1 = features[i]
            for j in context_index:
                feature_2 = features[j]
                feature = np.concatenate([feature_1, feature_2])
                context_features.append(feature)
            qs = compound_classifier.predict_proba(context_features)
            temporal_qs_list.append(qs)
        else:
            temporal_qs_list.append([])

    pps = np.zeros((probability.shape[0], probability.shape[1]))
    max_step = iteration_num
    prediction_set = set()
    for t in range(max_step):
        ps = []
        for i in range(len(spatial_context)):
            p_0 = probability[i, :]
            p_list = [p_0.reshape(1, -1) * self_weight]
            context_index = spatial_context[i]
            if len(context_index) > 0:
                p_1 = np.zeros((len(context_index), len(p_0)))
                qs = spatial_qs_list[i]
                for k in range(len(context_index)):
                    for j in range(len(p_0)):
                        compound_label = (method_class[j], method_class[predictions_1[context_index[k]]])
                        if compound_label_table.__contains__(compound_label):
                            index = compound_label_table[compound_label]
                            q = qs[k, index]
                        else:
                            q = p_0[j] * probability[context_index[k], predictions_1[context_index[k]]]
                        weight = 0
                        if spatial_transition_p.__contains__(compound_label):
                            weight = spatial_transition_p[compound_label]
                        p_1[k, j] = q * weight
                p_list.append(p_1*spatial_weight)
            context_index = temporal_context[i]
            if len(context_index) > 0:
                p_2 = np.zeros((len(context_index), len(p_0)))
                qs = temporal_qs_list[i]
                for k in range(len(context_index)):
                    for j in range(len(p_0)):
                        compound_label = (method_class[j], method_class[predictions_1[context_index[k]]])
                        if compound_label_table.__contains__(compound_label):
                            index = compound_label_table[compound_label]
                            q = qs[k, index]
                        else:
                            q = p_0[j] * probability[context_index[k], predictions_1[context_index[k]]]
                        weight = 0
                        if temporal_transition_p.__contains__(compound_label):
                            weight = temporal_transition_p[compound_label]
                        p_2[k, j] = q * weight
                p_list.append(p_2 * temporal_weight)

            sum_p = np.sum(np.vstack(p_list), axis=0)
            sum_p = np.minimum(sum_p, 200)
            sum_p = np.exp(sum_p)
            sum_p = sum_p / np.sum(sum_p)
            ps.append(sum_p)

        ps = np.array(ps)
        predictions_2 = np.argmax(ps, axis=1)
        if prediction_set.__contains__(tuple(predictions_2)) is True:
            break
        prediction_set.add(tuple(predictions_2))
        predictions_1 = predictions_2

    confidence = []
    for i in range(len(predictions_2)):
        method_index = int(predictions_2[i])
        confidence.append(ps[i, method_index])
    confidence = np.array(confidence)
    methods = np.array([method_class[int(i)] for i in predictions_2])
    return methods, confidence


def sample_sequence(p):
    r = np.random.random(p.shape[0])
    pp = np.cumsum(p,axis=1)
    sequence = []
    for i in range(len(r)):
        sequence.append(int(np.min(np.argwhere(pp[i,:]>=r[i]).flatten())))
    # sequence = np.array(sequence)
    return sequence


def get_test_method_list(sample_list):
    method_list = []
    for i in range(len(sample_list)):
        flow_table, _ = sample_list[i]
        for flow_name in flow_table.keys():
            groundtruth = flow_table[flow_name][0]
            for method in groundtruth:
                method_list.append(method)
        method_list = list(set(method_list))
    return method_list


def test(active_app, burst_interval, approach='greedy'):
    setup_file_name = model_folder + r'setup_' + active_app + '_' + str(burst_interval) + '.pkl'
    with open(setup_file_name, 'rb') as file:
        training_sample_file_list = pickle.load(file)

    testing_sample_file_list = []
    for index in range(max_sample_num):
        file_name = testing_app_data_path + 'sample_' + active_app + '_' + str(burst_interval) + '_' + str(index) + '.pkl'
        testing_sample_file_list.append(file_name)
    testing_sample_file_list = list(set(testing_sample_file_list)-set(training_sample_file_list))

    model_file_name = model_folder + r'crf_model_' + str(active_app) + '_' + str(burst_interval)+'.model'
    with open(model_file_name, 'rb') as file:
        unary_classifier, compound_classifier, spatial_transition_p, temporal_transition_p, compound_label_table = pickle.load(file)

    weight_file_name = model_folder + r'weight_' + str(active_app) + '_' + str(burst_interval)+'.model'
    if exists(weight_file_name) is True:
        global self_weight
        global spatial_weight
        global temporal_weight
        with open(weight_file_name, 'rb') as file:
            self_weight, spatial_weight, temporal_weight = pickle.load(file)
            print('self weight:'+str(self_weight)+', spatial weight:'+str(spatial_weight)+', temporal weight:'+str(temporal_weight))

    if spatial_weight == 1:
        spatial_weight = 1
    if temporal_weight == 1:
        temporal_weight = 1
    testing_data_list = load_data(testing_sample_file_list)
    sample_list = get_testing_sample(testing_data_list)

    method_class = unary_classifier.classes_
    method_table = {method_class[i]: i for i in range(len(method_class))}
    result = []

    for i in tqdm(range(len(sample_list))):
        flow_table, burst_samples = sample_list[i]
        if approach == 'greedy':
            methods, confidence = crf_greedy(burst_samples, unary_classifier, compound_classifier, spatial_transition_p, temporal_transition_p, compound_label_table)

        for flow_name in flow_table.keys():
            groundtruth = flow_table[flow_name][0]
            label = np.zeros(len(method_class))
            for method in groundtruth:
                if method_table.__contains__(method):
                    label[method_table[method]] = 1
            burst_index = flow_table[flow_name][1]

            prediction = np.zeros(len(method_class))
            for j in burst_index:
                if method_table.__contains__(methods[j]):
                    method_index = method_table[methods[j]]
                    prediction[method_index] = max(prediction[method_index], confidence[j])

            result.append([label, prediction])

    performance = evaluate_performance(result, method_class)
    if approach == 'greedy':
        print('FOAP, greedy crf precision: '+str(performance[0][1])+', recall: '+str(performance[0][2])+', F1-score:'+str(performance[0][3]))


def normalize_prediction(temp_prediction, confidence_threshold):
    temp_prediction = temp_prediction.tolist()
    for i in range(len(temp_prediction)):
        temp_prediction[i] = list(map(lambda x: 1 if x - confidence_threshold>0 else 0, temp_prediction[i]))
    return np.array(temp_prediction)


def evaluate_performance(result, method_class):
    confidence_threshold_list = np.arange(0,1,0.1).tolist()
    labels = [x[0] for x in result]
    labels = np.array(labels)
    predictions = [x[1] for x in result]
    prediction = np.array(predictions)
    performance = []
    for confidence_threshold in confidence_threshold_list:
        temp_prediction = copy.deepcopy(prediction)
        temp_prediction = normalize_prediction(temp_prediction, confidence_threshold)
        result_M = np.zeros((len(method_class), 4))
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] == 1 and temp_prediction[i, j] == 1:
                    result_M[j, 0] += 1
                elif labels[i, j] == 1 and temp_prediction[i, j] == 0:
                    result_M[j, 3] += 1
                elif labels[i, j] == 0 and temp_prediction[i, j] == 1:
                    result_M[j, 1] += 1
                elif labels[i, j] == 0 and temp_prediction[i, j] == 0:
                    result_M[j, 2] += 1
        precision = np.sum(result_M[:, 0]) / (np.sum(result_M[:, 0]) + np.sum(result_M[:, 1]))
        recall = np.sum(result_M[:, 0]) / (np.sum(result_M[:, 0]) + np.sum(result_M[:, 3]))
        if precision+recall == 0:
            F1 = 0
        else:
            F1 = 2*precision*recall/(precision+recall)
        performance.append([confidence_threshold, precision, recall, F1])
    return performance


def update_experiment_setup(active_app, burst_interval, train_sample_index):
    training_sample_file_list = []
    for index in train_sample_index:
        file_name = training_method_data_path+'sample_'+active_app+'_'+str(burst_interval)+'_'+str(index)+'.pkl'
        training_sample_file_list.append(file_name)
    setup_file_name = model_folder + r'setup_' + active_app + '_' + str(burst_interval) + '.pkl'
    if force_update_setup is True or exists(setup_file_name) is False:
        with open(setup_file_name, 'wb') as file:
            pickle.dump(training_sample_file_list, file)
            print('update experimental setup')


def train_and_test(active_app, burst_interval, is_train=True, is_test=True):
    try:
        if is_train is True:
            print('training stage:')
            if train(active_app, burst_interval) == -1:
                return
            train_weight(active_app, burst_interval)
        if is_test is True:
            print('testing stage:')
            test(active_app, burst_interval, approach='greedy')
    except Exception as exc:
        print(exc)


def list_app(sample_path):
    f = []
    for (dirpath, dirnames, filenames) in walk(sample_path):
        f.extend(filenames)
        break
    app_list = []
    for file_name in f:
        if file_name.__contains__('.pkl'):
            strs = file_name.split('_')
            if len(strs)<4:
                continue
            del strs[0]
            del strs[-1]
            del strs[-1]
            app_name = '_'.join(strs)
            app_list.append(app_name)
    app_list = list(set(app_list))
    return app_list


def get_sample_index(app_name, sample_index, burst_interval):
    temp_sample_index = []
    for index in sample_index:
        sample_file_name = training_method_data_path + 'sample_' + app_name + '_' + str(burst_interval) + '_' + str(
            index) + '.pkl'
        if exists(sample_file_name):
            temp_sample_index.append(index)
    return temp_sample_index


def run(app_index, app_list, burst_interval=0.5):
    app_name = app_list[app_index]
    sample_index = list(range(max_sample_num))
    sample_index = get_sample_index(app_name, sample_index, burst_interval)
    training_sample_num = int(np.round(len(sample_index)*training_ratio))
    random.shuffle(sample_index)
    train_sample_index = sample_index[:training_sample_num]
    test_sample_index = sample_index[training_sample_num:]

    print('-' * 100)
    update_experiment_setup(app_name, burst_interval, train_sample_index)
    print('app:' + app_name)
    print('burst interval:' + str(burst_interval))
    train_and_test(app_name, burst_interval, is_train=True, is_test=True)


if __name__ == '__main__':
    my_app_index = 10
    my_burst_interval = 0.5
    training_ratio = 0.8
    show_info = False

    # force_training = True
    force_training = False
    # force_update_setup = True
    force_update_setup = False
    force_testing = True
    # force_testing = False

    model_folder = r'E:/Paper/FOAP/Modify/foap_project/FOAP_model/'
    training_method_data_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_data/method_sample/'
    testing_app_data_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_data/method_sample/'

    my_app_list = list_app(training_method_data_path)

    run(my_app_index, my_app_list, my_burst_interval)

    print('Done!')
