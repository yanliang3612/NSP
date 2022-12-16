import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model.cnnrnn import CNNRNN
from src.args import parse_args
from src.utils import set_random_seeds,summary
from src.data import load_data,data_split
from torch.autograd import Variable

args = parse_args()


# checking if GPU is available
device = torch.device("cpu")
if (torch.cuda.is_available()):
    device = torch.device("cuda:0")
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')



def main():
    test_loss_list =[]
    for fold in range(args.repetitions):
        set_random_seeds(fold)
        data_x,data_y = load_data()
        train_x, train_y, test_x, test_y = data_split(data_x,data_y,args.trainsize)
        train_x, train_y, test_x, test_y = train_x.to(device), train_y.to(device), test_x.to(device), test_y.to(device)
        input_size_num = data_x.shape[1]
        output_size_num = 1
        CNNRNN_model = CNNRNN().to(device)
        print('CNNRNN model:', CNNRNN_model)
        print('model.parameters:', CNNRNN_model.parameters)
        print('train x tensor dimension:', Variable(train_x).size())

        # optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(CNNRNN_model.parameters(), lr=args.lr)

        prev_loss = 1000

        for epoch in range(args.epochs):
            output = CNNRNN_model(train_x).to(device)
            loss = criterion(output, train_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss < prev_loss:
                torch.save(CNNRNN_model.state_dict(), 'CNNRNN_model.pt')  # save model parameters to files
                prev_loss = loss

            if loss.item() < 1e-4:
                print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, args.epochs, loss.item()))
                print("The loss value is reached")
                break
            elif (epoch + 1) % 100 == 0:
                print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, args.epochs, loss.item()))


        # ----------------- test -------------------

        lstm_model = CNNRNN_model.eval()  # switch to testing model

        # prediction on test dataset


        output_test = CNNRNN_model(test_x).to(device)
        test_loss = criterion(output_test,test_y )
        # print("test loss：", test_loss.item())
        test_loss_list.append(test_loss.item())

    log = summary(test_loss_list,args.repetitions)
    print(log)


if __name__ == "__main__":
    main()