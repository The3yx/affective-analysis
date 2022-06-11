import argparse
from BERT_larget import BertModel
from data import getData
from LSTM import LSTM

def TODO():
    return


def main(args):
    #获取模型等操作
    device = 'cuda:5' #换成对应的GPU
    tokenizer = BertTokenizer.from_pretrained('./bert-large-uncased', do_lower_case=True)
    bert = BertModel.from_pretrained('./bert-large-uncased')
    bert.eval()
    bert.to(device)
    #所有训练集路径，测试集路径
    myPath=['./dataset/train_split.csv']
    pathVal='./dataset/eval_split.csv'
    filename="text" #部分文件名
    mask=False #设置是否使用mask，即是否取平均
    layers=False #保留所有层

    bertModel = BertModel()
    maxlen = bertModel()

    # 加载数据集
    trainPath="textTrainData"
    valPath="textValData"
    if mask:
        trainPath = 'mask' + trainPath
        valPath = 'mask' + valPath

    trainData, testData = getData()
    trainLen=len(trainData)
    testLen=len(testData)
    print(trainLen)
    print(testLen)
    trainDataLoader=DataLoader(trainData,batch_size=batch_size,shuffle=True,drop_last=True)
    testDataLoader=DataLoader(testData,batch_size=batch_size,shuffle=True,drop_last=True)
    # 初始化MODEL
    if mask:
        newModel =LSTMmask(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    else:
        newModel =LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    # 训练
    newModel=newModel.to(device)

    theLoss=nn.CrossEntropyLoss()
    theLoss=theLoss.to(device)

    optim =torch.optim.Adam(newModel.parameters(), lr=lr)

    epoch=360
    write =SummaryWriter(logPath)
    theStep=0
    for i in range(epoch):
        if i%30==0:
            lr = lr*0.7
            optim = torch.optim.Adam(newModel.parameters(), lr=lr)
        newLoss=0
        acc=0
        theStep+=1
        newModel.train()
        for data in trainDataLoader:
            wav,flag=data
            flag = flag.reshape(-1)
            wav=wav.reshape(-1,1,input_dim)
            wav=wav.to(device)
            flag=flag.to(device)
            output=newModel(wav)
            output = output.reshape(batch_size, -1)
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
                wav = wav.reshape(-1, 1, input_dim)
                flag=torch.tensor(flag)
                wav=wav.to(device)
                flag=flag.to(device)
                output=newModel(wav)
                output = output.reshape(batch_size, -1)
                loss=theLoss(output,flag)
                newLoss+=loss
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
    parser.add_argument('--hidden_dim', type=int, default=64, help='Dimension of the hidden layer.')
    parser.add_argument('--is_mask', type=bool, default=False, help='Whether use the mask method.')
    parser.add_argument('--drop_rate', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--num_layers', type=int, default=1, help='The numbers of the layers in down stream model.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')

    
    args = parser.parse_args()

    main(args)
