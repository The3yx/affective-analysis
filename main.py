import argparse

def TODO():
    return


def main(args):
    TODO()
    #加载数据集
    





    #初始化MODEL






    #训练








    #Test
    
    return












if __name__ == '__main__':

    #加载超参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Dimension of the hidden layer.')
    parser.add_argument('--is_mask', type=bool, default=False, help='Whether use the mask method.')
    parser.add_argument('--drop_rate', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
    
    args = parser.parse_args()

    main(args)