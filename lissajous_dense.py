import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from matplotlib import pyplot as plt


class Model (nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.l1 = nn.Linear(6,20)
    self.l2 = nn.Linear(20,40)
    self.l3 = nn.Linear(40,6)

  def forward(self,x):
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    x = self.l3(x)
    return x

class my_dataset(torch.utils.data.Dataset):
  def __init__(self, point, noise):

    self.point = point
    self.len = len(point)  #np.arrayならpoint.shape[0]でも可
    self.noise = noise

  def __getitem__(self, index):

    if self.noise == True:
        input = self.point[index-1] + np.random.normal(scale=0.001) #indexの扱いに気を付ける
        target = self.point[index]  + np.random.normal(scale=0.001) #最初と同じ点にならないように
    else:
        input = self.point[index-1]  #indexの扱いに気を付ける
        target = self.point[index]   #最初と同じ点にならないように
    return input, target

  def __len__(self):
    return self.len

def get_dataset(n):

    for i in range (1,5):
      if i == 1:

        parameter = np.radians(np.linspace(1, 90, int(n/4)))
        x_1 = 2*np.sin(parameter)
        y_1 = np.sin(2*parameter)
        dataset_1 = np.array([x_1, y_1, np.ones(int(n/4)), np.zeros(int(n/4)), np.zeros(int(n/4)), np.zeros(int(n/4))])

      elif i == 2:

        parameter = np.radians(np.linspace(91, 180, int(n/4)))
        x_2 = 2*np.sin(parameter)
        y_2 = np.sin(2*parameter)
        dataset_2 = np.array([x_2, y_2, np.zeros(int(n/4)), np.ones(int(n/4)), np.zeros(int(n/4)), np.zeros(int(n/4))])

      elif i == 3:

        parameter = np.radians(np.linspace(181, 270, int(n/4)))
        x_3 = 2*np.sin(parameter)
        y_3 = np.sin(2*parameter)      
        dataset_3 = np.array([x_3, y_3, np.zeros(int(n/4)), np.zeros(int(n/4)), np.ones(int(n/4)), np.zeros(int(n/4))])

      else:

        parameter = np.radians(np.linspace(271, 360, int(n/4)))
        x_4 = 2*np.sin(parameter)
        y_4 = np.sin(2*parameter)
        dataset_4 = np.array([x_4, y_4, np.zeros(int(n/4)), np.zeros(int(n/4)), np.zeros(int(n/4)), np.ones(int(n/4))])      

    dataset = np.concatenate([dataset_1, dataset_2, dataset_3, dataset_4], 1)
    x = np.concatenate([x_1, x_2, x_3, x_4])
    y = np.concatenate([y_1, y_2, y_3, y_4])

    return dataset, x, y

def train(data_loader, model, criterion, optimizer, epoch):
    
    train_loss_list = []
    for t in range(epoch):

        train_loss = 0
        for batch_data in data_loader: # 1ミニバッチずつ計算

            input, target = batch_data
            optimizer.zero_grad()

            output = model(input.float())

            loss = criterion(output, target.float())
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            print(t, loss.item())
        ave_train_loss = train_loss/len(data_loader.dataset)
        train_loss_list.append(ave_train_loss)

        print("ave:" ,ave_train_loss)
        print()
    drawing_loss_graph(epoch, train_loss_list)

def test(first_point, model):

    record_output = np.array([[0.0, 0.0]])

    for i in range(360):
        test_output = model(first_point)
        first_point = test_output
        test_output = test_output.reshape(1,2).detach().numpy()
        record_output = np.concatenate(([record_output, test_output]), axis=0)

    return record_output

def drawing_loss_graph(epoch, train_loss_list):
 
    loss_fig = plt.figure()
    plt.plot(range(epoch), train_loss_list, linestyle='-', label='train_loss')

    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.show()

def main():

    data_number = 360 #準備するデータの数
    batch_size  = 20 # 1つのミニバッチのデータの数

    points, x, y = get_dataset(data_number)
    dataset = my_dataset(points, noise=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    model = Model()
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_log = [] # 学習状況のプロット用
    epoch = 100

    first_point = [[0.0, 0.0]]
    first_point = torch.Tensor(first_point)

    model.train()
    train(data_loader, model, criterion, optimizer, epoch)

    model.eval()
    final_test_output = test(first_point, model)

    final_x, final_y = final_test_output.T

    print("go")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    plt.plot(x, y, label="original", color = "y")
    plt.scatter(final_x, final_y, label="trained", color = "b")
if __name__ == '__main__':
    main()
