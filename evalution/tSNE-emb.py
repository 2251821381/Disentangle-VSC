import os

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def get_data(input_path):
    _, _, fileList = next(os.walk(input_path))
    fileList.sort()
    dataList = []
    labelList = []
    for fileName in sorted(fileList):
        label = fileName.split('_')[0]
        emb = np.load(os.path.join(input_path, fileName))
        dataList.append(emb)
        labelList.append(label)
    return dataList, labelList


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    spk = ['p225', 'p226', 'p227', 'p228', 'p229', 'p230', 'p231', 'p232', 'p233', 'p234']
    mapping = dict([(v, str(i)) for i, v in enumerate(list(set(spk)))])
    for i in mapping.keys():
        indexes = np.where(np.array(spk) == i)[0]
        org_indexes = indexes + len(spk)
        plt.xlim(np.min(data[:, 0]) - 25, np.max(data[:, 0]))
        ax.scatter(data[indexes, 0], data[indexes, 1], c='C' + mapping[i], s=15, label=i, marker='x')
        ax.scatter(data[org_indexes, 0], data[org_indexes, 1], c='C' + mapping[i], s=15, label=i, marker='^')


def main():
    data_dir = r'../models/audios/embeds'
    dataList, labelList = get_data(data_dir)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(dataList)
    plot_embedding(result, labelList, 't-SNE embedding of the digits')


if __name__ == '__main__':
    main()