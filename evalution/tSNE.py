# 首先为了省事，将训练神经网络得库全都导入进来
import os

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


# # 加载数据
# def get_data():
#     """
#     @return: 数据集、标签、样本数量、特征数量
#     """
#     # digits = datasets.load_digits(n_class=6)
#     # data = digits.data
#     # label = digits.target
#     # n_samples, n_features = data.shape
#
#     # 载入数据
#     df = pd.read_csv(r'C:/Users/1/Desktop/4改.csv')
#     X = np.expand_dims(df.values[:, 0:1024].astype(float), axis=2)
#     Y = df.values[:, 1024]
#     # 这里很重要，不把标签做成独热码得形式最终出的图空白一片
#     encoder = LabelEncoder()
#     Y_encoded = encoder.fit_transform(Y)
#     Y_onehot = np_utils.to_categorical(Y_encoded)
#     X = X.reshape(8621, 1024)  # 这个形状根据读取到得csv文件得的大小自行调试，8621是列数，此列数比CSV文件中少一行
#     data = X  # digits.data		# 图片特征
#     label = Y  # digits.target		# 图片标签
#     n_samples = 8621  # 对应reshape中的行数
#     n_features = 1024  # 对应reshape中的列数
#     return data, label, n_samples, n_features

def get_data(input_path, label):
    """
    读取input_path下的声谱图，并给每张图打上自定义标签
    @param input_path:
    @param label:
    @return:
    """
    segment_size = 128
    fileList = os.listdir(input_path)
    fileList.sort(key=lambda x: int(x.split('_')[1]))
    data_list = np.zeros((len(fileList), 128 * 80))
    label_list = []

    # 为当前文件下所有图片分配自定义标签label
    for k in range(len(fileList)):
        label_list.append(label)

    for i in range(len(fileList)):
        image_path = os.path.join(input_path, fileList[i])
        mel = np.load(image_path)
        # 截取图片，(128, 80)
        if len(mel) > segment_size:
            max_start = len(mel) - segment_size
            left = np.random.randint(0, max_start)
        else:
            left = 0
        mel = mel[left:left + segment_size, :]
        # 使得a的数据都在[0, 1]区间内，即小于0的数字改成0，大于1的改成1
        mel = np.clip(mel, 0, 1)
        # 用0填充a，使得第一维大小为segment_size=128
        mel = np.pad(mel, ((0, segment_size - len(mel)), (0, 0)), 'constant')
        mel = mel.flatten()
        data_list[i] = mel
    return data_list, label_list


# 对样本进行预处理并画图
def plot_embedding(data, label, title):
    """

    @param data: 数据集
    @param label: 样本标签
    @param title: 图像标题
    @return: 图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理

    fig = plt.figure()  # 创建图形实例
    ax = plt.subplot(111)  # 创建子图，经过验证111正合适，尽量不要修改
    spk = ['p225', 'p226', 'p227', 'p228', 'p229', 'p230', 'p231', 'p232', 'p233', 'p234']
    mapping = dict([(v, str(i)) for i, v in enumerate(list(set(spk)))])

    # ids = {'p225': 0, 'p226': 1, 'p227': 2, 'p228':3, 'p229': 4, 'p230': 5, 'p231': 6, 'p232': 7, 'p233': 8, 'p234': 9}
    # 遍历所有样本
    i = 0
    while i < data.shape[0]:
        x = []
        y = []
        label_id = label[i][0]
        for k in range(50):
            x.append(data[i, 0])
            y.append(data[i, 1])
            i = i + 1
        # 在图中为每个数据点画出标签
        plt.scatter(x, y, marker="x", color=plt.cm.Set1(mapping[label_id]), label=label_id)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
    # 返回值
    return fig


# 主函数，执行t-SNE降维
def main():
    dataset_dir = r'../dataset/VCTK/dataset/spmel'
    dirName, subdirList, _ = next(os.walk(dataset_dir))

    dataList = []
    labelList = []
    i = 0
    for subdir in sorted(subdirList):
        if i == 10:
            break
        # speaker, melList, _ = next(os.walk(subdir))
        melList = os.path.join(dataset_dir, subdir)
        print('Producing %s', subdir)
        data_tmp, label_tmp = get_data(melList, subdir)
        dataList.append(data_tmp)
        labelList.append(label_tmp)
        i = i + 1
    data = np.vstack(dataList)
    label = np.vstack(labelList)
    label = label.reshape(500, -1)
    print('the shape of data is %s, the shape of label is %s' % (data.shape, label.shape))
    # 调用函数，获取数据集信息
    # data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    # t-SNE降维
    result = tsne.fit_transform(data)
    # 调用函数，绘制图像
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits')
    # 显示图像
    # plt.show(fig)
    # #显示图片
    # plt.imshow(fig)
    # fig.show()

# 主函数
if __name__ == '__main__':
    main()
