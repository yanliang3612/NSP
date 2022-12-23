import torch
# Since we have very limited data, we use data augmentation methods such as Self-Training, Over-sampling, and SMOTE algorithms to train more robust models.



#SMOTE for regression

def smote_regression(x,y,device):
    tensor_new_x =  torch.empty(1,192).to(device)
    tensor_new_y = torch.empty(1,1).to(device)
    num = x.shape[0]
    for i in range(num):
        for j in (range(num))[i+1:]:
            new_x = ((x[i] + x[j]) * 0.5).reshape(1,192)
            new_y = ((y[i] + y[j]) * 0.5).reshape(1,1)
            tensor_new_x = torch.cat((tensor_new_x, new_x), 0)
            tensor_new_y = torch.cat((tensor_new_y, new_y), 1)
    new_x_train = torch.cat((tensor_new_x[1:], x), 0)
    new_y_train = (torch.cat((tensor_new_y[:,1:], y.reshape(1,8)), 1)).reshape(36)
    return new_x_train,new_y_train






#SMOTE for classification












# Oversampling for regression and classification

def oversampling(x,y,num_argmen,device):
    tensor_new_x =  torch.empty(1,192).to(device)
    tensor_new_y = torch.empty(1,1).to(device)
    num = x.shape[0]
    for i in range(num):
        for j in range(num_argmen):
            new_x = ((x[i] + x[i]) * 0.5).reshape(1,192)
            new_y = ((y[i] + y[i]) * 0.5).reshape(1,1)
            tensor_new_x = torch.cat((tensor_new_x, new_x), 0)
            tensor_new_y = torch.cat((tensor_new_y, new_y), 1)
    new_x_train = torch.cat((tensor_new_x[1:], x), 0)
    new_y_train = (torch.cat((tensor_new_y[:,1:], y.reshape(1,8)), 1)).reshape(8+num_argmen*num)
    return new_x_train,new_y_train


