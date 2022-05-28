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
