import numpy as np
import torch
from torch import nn
import random
import pandas as pd
import pickle

batch_size = 100
p_threshold = 0.5
detection_window = 10
# warm_start = True
warm_start = False
configure_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_model/training_configure.txt'
model_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_model/flow_embedding_1.model'
learning_rate = 0.01
training_class = 2
chosen_num = 50
raw_data_path = r'E:/Paper/FOAP/Modify/foap_project/FOAP_data/metric_learning_sample/'


class FlowEmbedding(nn.Module):
    def __init__(self, feature_num):
        super(FlowEmbedding,self).__init__()
        self.input_size = feature_num
        self.hidden_size = 100
        self.output_size = 5

        self.dropout_rate = 0.1

        self.scale = nn.Parameter(torch.abs(torch.randn(1, dtype=torch.float, requires_grad=True)))

        self.net = nn.Sequential(nn.Linear(self.input_size,self.hidden_size),
                                 nn.Dropout(self.dropout_rate),
                                 nn.PReLU(),
                                 nn.Linear(self.hidden_size, self.hidden_size),
                                 nn.Dropout(self.dropout_rate),
                                 nn.PReLU(),
                                 nn.Linear(self.hidden_size, self.output_size),
                                 nn.BatchNorm1d(self.output_size)
        )

    def forward(self, x):
        z = self.net(x)
        z = z*self.scale
        return z


def load_data(app_list):
    data_path = raw_data_path
    data_list = []
    data_table = {}
    sample_index = list(range(5))

    for index in sample_index:
        for app_label in app_list:
            file_name = data_path + str(app_label) + '_' + str(index) + '.txt'
            X = np.array(pd.read_csv(file_name, header=None))
            labels = X[:,2].tolist()
            X = X[:, 4:].tolist()
            for i in range(len(labels)):
                label = labels[i]
                if data_table.__contains__(label) is False:
                    data_table[label] = []
                data_table[label].append(torch.from_numpy(np.array(X[i],dtype='float32')))

    for label in data_table.keys():
        if len(data_table[label])>1:
            data_list.append(data_table[label])

    return data_list


def get_sample_similarity(sample_1, sample_2, identical):
    sample_1 = torch.stack(sample_1)
    sample_2 = torch.stack(sample_2)
    m_1 = sample_1.shape[0]
    n = sample_1.shape[1]
    m_2 = sample_2.shape[0]
    ss = []
    for i in range(m_1):
        s = []
        for j in range(m_2):
            if identical is True and i == j:
                pass
            else:
                s.append(torch.norm(sample_1[i, :] - sample_2[j, :]))
        ss.append(torch.min(torch.stack(s)))
    ss = torch.stack(ss)
    return ss


def batch_training(data_subset):
    mapper.train()

    index_list = []
    sample_list = []
    batch_data = []
    for i in range(len(data_subset)):
        data = data_subset[i]
        random.shuffle(data)
        batch_data.append([])
        for j in range(len(data)):
            trace = data[j]
            batch_data[i].append(1)
            sample_list.append(trace)
            index_list.append((i, j))
    zs = mapper(torch.stack(sample_list))

    for t in range(len(index_list)):
        i,j = index_list[t]
        batch_data[i][j] = zs[t,:]

    inputs = []
    targets = []
    for i in range(len(batch_data)):
        sample_0 = batch_data[i]
        ds = []
        for j in range(training_class):
            sample_1 = batch_data[i-j]
            if j == 0:
                identical = True
            else:
                identical = False

            similarity_1 = get_sample_similarity(sample_0, sample_1, identical)
            d_1 = -similarity_1
            ds.append(d_1)
        ds = torch.stack(ds,dim=1)
        inputs.append(ds)
        target = [0 for _ in range(len(sample_0))]
        targets.extend(target)

    inputs = torch.cat(inputs, dim=0)
    targets = torch.from_numpy(np.array(targets))
    targets = targets.long()
    return inputs, targets


def test_performance(data_list_test):
    mapper.eval()

    random.shuffle(data_list_test)
    data_subset = data_list_test[:chosen_num]
    index_list = []
    sample_list = []
    batch_data = []
    for i in range(len(data_subset)):
        data = data_list_test[i]
        random.shuffle(data)
        batch_data.append([])
        for j in range(len(data)):
            trace = data[j]
            batch_data[i].append(1)
            sample_list.append(trace)
            index_list.append((i, j))
    zs = mapper(torch.stack(sample_list))

    for t in range(len(index_list)):
        i, j = index_list[t]
        batch_data[i][j] = zs[t, :]

    inputs = []
    targets = []
    for i in range(len(batch_data)):
        sample_0 = batch_data[i]
        ds = []
        for j in range(training_class):
            sample_1 = batch_data[i - j]
            if j == 0:
                identical = True
            else:
                identical = False

            similarity_1 = get_sample_similarity(sample_0, sample_1, identical)
            # d_1 = torch.log(torch.clamp_min_(similarity_1, 1e-20))
            d_1 = -similarity_1
            ds.append(d_1)
        ds = torch.stack(ds, dim=1)
        inputs.append(ds)
        target = [0 for _ in range(len(sample_0))]
        targets.extend(target)

    inputs = torch.cat(inputs, dim=0)
    targets = torch.from_numpy(np.array(targets))
    prediction = torch.argmax(inputs, dim=1)
    counter_1 = 0
    counter_2 = 0
    for i in range(len(prediction)):
        if prediction[i] != targets[i]:
            counter_1 += 1
        counter_2 += 1
    error_rate = round(float(counter_1) / counter_2, 2)
    return error_rate


def train(data_list_train, data_list_test):
    mapper.train()
    learning_rate = 0.01
    optimizer = torch.optim.Adam(mapper.parameters(), lr=learning_rate)
    loss_f = nn.CrossEntropyLoss()

    repeat_num = 50000

    for k in range(repeat_num):
        counter_1 = 0
        counter_2 = 0
        random.shuffle(data_list_train)
        data_subset = data_list_train[:chosen_num]
        inputs, targets = batch_training(data_subset)
        prediction = torch.argmax(inputs, dim=1)
        for i in range(len(prediction)):
            if prediction[i] != targets[i]:
                counter_1 += 1
            counter_2 += 1

        loss = loss_f(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = loss.detach().numpy()

        if k % 10 == 0:
            save_model()
            # mapper.eval()
            test_error_rate = test_performance(data_list_test)
            print(str(k) + ': loss is ' + str(total_loss) + ', training error rate: ' + str(round(float(counter_1) / counter_2, 2))+', testing error rate:'+str(test_error_rate))


def save_model():
    torch.save({
        'model_state_dict': mapper.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)


def load_model():
    checkpoint = torch.load(model_path)
    mapper.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def performance_evaluation(prediection, labels):
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    for i in range(len(labels)):
        if labels[i] == 1 and prediection[i] == labels[i]:
            TP += 1
        if labels[i] == 1 and prediection[i] != labels[i]:
            FN += 1
        if labels[i] == 0 and prediection[i] == labels[i]:
            TN += 1
        if labels[i] == 0 and prediection[i] != labels[i]:
            FP += 1
    if TP+FP == 0:
        precision = 0
    else:
        precision = float(TP)/(TP+FP)
    if TP+FN == 0:
        recall = 0
    else:
        recall = float(TP) / (TP + FN)

    precision = round(precision,2)
    recall = round(recall,2)
    return precision, recall


if __name__ == '__main__':
    active_app = 0
    app_number = 100
    other_app_num = 5
    training_index = list(range(3))
    testing_index = list(range(3, 5))

    if warm_start is False:
        other_apps = [i for i in range(app_number)]
        other_apps = list(filter(lambda x: x != active_app, other_apps))
        random.shuffle(other_apps)
        other_apps_1 = other_apps[:other_app_num]
        random.shuffle(other_apps)
        other_apps_2 = other_apps[other_app_num:other_app_num*2]
        other_apps_3 = other_apps[other_app_num*2:]

        training_configure = []
        training_configure.append(active_app)
        training_configure.append(other_apps_1)
        training_configure.append(other_apps_2)
        training_configure.append(other_apps_3)

        with open(configure_path,'wb') as file:
            pickle.dump(training_configure, file)

    else:
        with open(configure_path,'rb') as file:
            training_configure = pickle.load(file)
        active_app = training_configure[0]
        other_apps_1 = training_configure[1]
        other_apps_2 = training_configure[2]
        other_apps_3 = training_configure[3]

    data_list = load_data(other_apps_3)
    random.shuffle(data_list)
    n = int(len(data_list)*0.6)
    training_data_list = data_list[:n]
    testing_data_list = data_list[n:]

    feature_num = training_data_list[0][0].shape[0]

    if warm_start is False:
        mapper = FlowEmbedding(feature_num)
        optimizer = torch.optim.Adam(mapper.parameters(), lr=learning_rate)
    else:
        mapper = FlowEmbedding(feature_num)
        optimizer = torch.optim.Adam(mapper.parameters(), lr=learning_rate)
        load_model()

    train(training_data_list, testing_data_list)

    print('Done!')
