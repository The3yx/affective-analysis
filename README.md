# affective-analysis

2022BUPT机器学习大作业

处理数据就还是变成csv存起来

预训练模型在https://huggingface.co/下载

git clone https://huggingface.co/bert-large-uncased

下载方式如上所示

预训练请都使用larget和英文的模型

预训练需要输出两个数据，一个是取平均的，一个是不取平均的存储格式如下，不取平均的需要输出特征的最大值，特征维度也需要告诉下一级（可以看bert的代码）

预训练模型的输出按1:4分成训练集合数据集分别存储，文件名是所在行+“.npy”

![1~5V`0)I_VT5W9(KP}16LE1](https://user-images.githubusercontent.com/72617488/170818901-65fb4783-0c12-49a0-b726-cfc37d8eab40.png)

transform就在lstm的代码上改就可以了吧，换掉model其他部分应该不用改，mask可以直接迁移过去，可能要调整一下维度的位置（比如时间维度在第一个或第二个之类的）

模型思路的话没啥思路吧，之前这么做都是比较好的

因为数据太少希望可以数据增强获得更多的数据，处理杂音就肯定需要的（处理错误标签？）

因为数据太少，所以需要使用预训练模型进行提取特征，以提高训练的成功率，bert是最经典的模型，后两者在他的基础上改进，是最近的sota

预训练之后需要对时序信息再进行一波处理，rnn，lstm，transformer都是可以的选择，一般来说后两者比较好，所以这里使用了后两者

mask的使用是因为输入的大小不同，可以取平均或者mask


数据不平衡：60%的3
解决方案：

1. 数据增强其他类型
2. [(30条消息) 视觉分类任务中处理不平衡问题的loss比较_Daniel2333的博客-CSDN博客](https://blog.csdn.net/weixin_35653315/article/details/78327408)
