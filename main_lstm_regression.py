import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model.lstm import LstmRNN
from src.args import parse_args
from src.utils import set_random_seeds,summary_mse,summary_mae
from src.data import load_data_regression,load_data_regression_more,data_split
from torch.autograd import Variable
from copy import deepcopy
from src.data_augmentation import smote_regression,oversampling
args = parse_args()




# checking if GPU is available
device = torch.device("cpu")

if (torch.cuda.is_available()):
    device = torch.device("cuda:0")
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')



def main():
    test_mse_loss_list =[]
    test_mae_loss_list = []
    for fold in range(args.repetitions):
        set_random_seeds(fold)
        if args.parside:
            data_x,data_y = load_data_regression_more()
        else:
            data_x, data_y = load_data_regression()
        train_x, train_y, test_x, test_y = data_split(data_x,data_y,args.trainsize)
        train_x, train_y, test_x, test_y = train_x.to(device), train_y.to(device), test_x.to(device), test_y.to(device)
        run_train_x,run_train_y = deepcopy(train_x),deepcopy(train_y)


        if args.smote:
            run_train_x, run_train_y = smote_regression(run_train_x,run_train_y,device)


        if args.oversampling:
            run_train_x, run_train_y = oversampling(run_train_x, run_train_y,args.numargmen,device)


        input_size_num = data_x.shape[1]
        output_size_num = 1
        LSTM_model = LstmRNN(input_size=input_size_num, hidden_size=args.hidden, output_size=output_size_num, num_layers=args.layers).to(device)
        print('LSTM model:', LSTM_model)
        print('model.parameters:', LSTM_model.parameters)
        print('train x tensor dimension:', Variable(run_train_x).size())

        # optimizer
        criterion = nn.MSELoss()
        criterion_more = nn.L1Loss()
        optimizer = torch.optim.Adam(LSTM_model.parameters(), lr=args.lr)

        prev_loss = 1000

        for epoch in range(args.epochs):
            output = LSTM_model(run_train_x).to(device)
            loss = criterion(output, run_train_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss < prev_loss:
                torch.save(LSTM_model.state_dict(), 'CNNRNN_model.pt')  # save model parameters to files
                prev_loss = loss

            if loss.item() < 1e-4:
                print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, args.epochs, loss.item()))
                print("The loss value is reached")
                break
            elif (epoch + 1) % 100 == 0:
                print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, args.epochs, loss.item()))


        # ----------------- test -------------------

        lstm_model = LSTM_model.eval()  # switch to testing model

        # prediction on test dataset

        testsize = test_y.shape[0]
        output_test = lstm_model(test_x).to(device)
        test_mse_loss = criterion(output_test,test_y )
        test_mae_loss = criterion_more(output_test,test_y)
        test_mse_loss_list.append(test_mse_loss.item())
        test_mae_loss_list.append(test_mae_loss.item())

    log_mse = summary_mse(test_mse_loss_list,args.repetitions,testsize)
    log_mae = summary_mae(test_mae_loss_list, args.repetitions,testsize)


    print(log_mse),print(log_mae)


if __name__ == "__main__":
    main()








