import argparse
from BERT_larget import MyBertModel
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertModel
from data import getData
from LSTM import LSTM
from utils import CreateFile
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import torch
from LSTM import LSTM, LSTMmask
from torch.utils.tensorboard import SummaryWriter 
from transformer import MyTransformer


def TODO():
    return


def main(args):
    #获取模型等操作
    device = 'cpu'
    tokenizer = BertTokenizer.from_pretrained('./bert-large-uncased', do_lower_case=True)
    bert = BertModel.from_pretrained('./bert-large-uncased')
    bert.eval()
    bert.to(device)
    #所有训练集路径，测试集路径
    myPath=["./dataset/clean_ori_data_train_split.csv"]
    pathVal='./dataset/clean_ori_data_test_split.csv'
    filename="text" #部分文件名
    mask=False #设置是否使用mask，即是否取平均
    layers=False #保留所有层
    CreateFile(filename)

    bertModel = MyBertModel(myPath, bert)
    maxlen = bertModel(filename = filename, tokenizer = tokenizer, pathVal = pathVal)
    # 加载数据集
    trainPath="textTrainData"
    valPath="textValData"
    if mask:
        trainPath = 'mask' + trainPath
        valPath = 'mask' + valPath

    trainData, testData = getData(myPath, pathVal, valPath, trainPath, maxlen)
    trainLen=len(trainData)
    testLen=len(testData)
    print(trainLen)
    print(testLen)
    trainDataLoader=DataLoader(trainData,batch_size=args.batch_size,shuffle=True,drop_last=True)
    testDataLoader=DataLoader(testData,batch_size=args.batch_size,shuffle=True,drop_last=True)
    # 初始化MODEL
    if mask:
        newModel =LSTMmask(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers)
    else:
        newModel =LSTM(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers)
        #newModel = MyTransformer(input_dim = args.input_dim)

    # 训练
    newModel=newModel.to(device)

    theLoss=nn.CrossEntropyLoss()
    theLoss=theLoss.to(device)
    lr = args.lr
    optim =torch.optim.Adam(newModel.parameters(), lr=lr,weight_decay=1e-5)

    epoch=360
    write =SummaryWriter('./logPath')
    theStep=0
    for i in range(epoch):
        if i%30==0:
            lr = lr*0.7
            optim = torch.optim.Adam(newModel.parameters(), lr=lr,weight_decay=1e-5 )
        newLoss=0
        acc=0
        theStep+=1
        newModel.train()
        for data in trainDataLoader:
            wav,flag=data
            flag = flag.reshape(-1)
            wav=wav.reshape(-1,1,args.input_dim)
            wav=wav.to(device)
            flag=flag.to(device)
            output=newModel(wav)
            output = output.reshape(args.batch_size, -1)
            loss=theLoss(output,flag)
            optim.zero_grad()
            loss.backward()
            optim.step()
            acc+=(output.argmax(1)==flag).sum()
            newLoss+=loss
            del wav,flag,loss

        print("训练集："+str(i) + "   loss:" + str(newLoss) + "   acc:" + str(acc / trainLen))
        write.add_scalar("训练loss",newLoss,theStep)
        write.add_scalar("训练acc",acc/trainLen,theStep)

        # Test
        acc=0
        newLoss=0
        newModel.eval()
        for data in testDataLoader:
            with torch.no_grad():
                wav,flag=data
                flag=flag.reshape(-1)
                wav = wav.reshape(-1, 1, args.input_dim)
                flag=torch.tensor(flag)
                wav=wav.to(device)
                flag=flag.to(device)
                output=newModel(wav)
                output = output.reshape(args.batch_size, -1)
                loss=theLoss(output,flag)
                newLoss+=loss
                print(output.argmax(1),flag)
                acc+=(output.argmax(1)==flag).sum()
        print("测试集："+str(i) + "   loss:" + str(newLoss) + "   acc:" + str(acc / testLen))
        write.add_scalar("loss",newLoss,theStep)
        write.add_scalar("acc",acc/testLen,theStep)
    write.close()
    
    return


if __name__ == '__main__':

    #加载超参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--input_dim', type=int, default=1024, help='Dimension of the input.')
    parser.add_argument('--hidden_dim', type=int, default=20, help='Dimension of the hidden layer.')
    parser.add_argument('--output_dim', type=int, default=5, help='Dimension of the output.')
    parser.add_argument('--is_mask', type=bool, default=False, help='Whether use the mask method.')
    parser.add_argument('--drop_rate', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--num_layers', type=int, default=1, help='The numbers of the layers in down stream model.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')

    args = parser.parse_args()

    main(args)
