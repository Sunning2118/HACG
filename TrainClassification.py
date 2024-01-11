
import pickle
from dgl.nn.pytorch.conv import RelGraphConv
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch import GraphConv, HeteroGraphConv, SAGEConv
from torch.utils.data import DataLoader, Dataset
import tqdm

class CodeDataSet(Dataset):
    def __init__(self, path):

        with open(path, 'rb') as fp:
            df_codeid_classid_graphds_all = pickle.load(fp)
            self.codeid_col_list = df_codeid_classid_graphds_all.iloc[:, 0].tolist()
            self.classid_col_list = df_codeid_classid_graphds_all.iloc[:, 1].tolist()
            self.graphds_col_list = df_codeid_classid_graphds_all.iloc[:, 2].tolist()
            self.etype_col_list = df_codeid_classid_graphds_all.iloc[:, 3].tolist()
            self.Node_feature_col_list = df_codeid_classid_graphds_all.iloc[:, 4].tolist()
    def __getitem__(self, index):
      return self.codeid_col_list[index], self.classid_col_list[index], self.graphds_col_list[index], self.etype_col_list[index], self.Node_feature_col_list[index]

    def __len__(self):
        return len(self.graphds_col_list)

def collate(samples):

    codeid, classid, graphds, etype, Node_feature = map(list, zip(*samples))
    batch_graph = dgl.batch(graphds)
    return batch_graph, torch.tensor([item for sublist in etype for item in sublist]), torch.cat(Node_feature, dim=0), torch.tensor(classid)


class Classifier(nn.Module):
    def __init__(self):

        super(Classifier, self).__init__()
        self.conv1 = RelGraphConv(128, 256, 3, regularizer='basis', num_bases=2)
        self.conv2 = RelGraphConv(256, 512, 3, regularizer='basis', num_bases=2)
        self.conv3 = RelGraphConv(512, 256, 3, regularizer='basis', num_bases=2)
        self.dropout = nn.Dropout(0.2)
        self.classify = nn.Linear(256, 250)
      
    def forward(self, g, f, e):

        
        f = F.relu(self.conv1(g, f, e))
        f = self.dropout(f)

        f = F.relu(self.conv2(g, f, e))
        f = self.dropout(f)

        f = F.relu(self.conv3(g, f, e))
        f = self.dropout(f)

        g.ndata['n_feature']=f
        f = dgl.mean_nodes(g, 'n_feature')

        f = self.classify(f)

        return f


if __name__ == '__main__':

    df_Codeid_Classid_GraphDS_train_path = "../data/java250/ForModel/df_codeid_classid_heteroGraphDS_sota_train.pkl"
    df_Codeid_Classid_GraphDS_test_path = "../data/java250/ForModel/df_codeid_classid_heteroGraphDS_sota_test.pkl"

    codeDataSet_train = CodeDataSet(df_Codeid_Classid_GraphDS_train_path)
    dataloader_train = DataLoader(dataset=codeDataSet_train, batch_size=32, shuffle=True, collate_fn=collate)

    codeDataSet_test = CodeDataSet(df_Codeid_Classid_GraphDS_test_path)
    dataloader_test = DataLoader(dataset=codeDataSet_test, batch_size=32, shuffle=True, collate_fn=collate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Classifier().to(device)

    loss_func = nn.CrossEntropyLoss()
    optimier = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    epoch_loss_list = []
    epoch_acc_list = []
    for epoch in range(100):
        epoch_loss = 0
        count = 0
        for iter, (bg, etype, feature, label) in enumerate(dataloader_train):
            count += 1
            bg = bg.to(device)
            etype = torch.tensor(etype).to(device)
            feature = torch.tensor(feature).to(device)
            label = label.to(device)
            
            prediction = model(bg, feature, etype)
          
            loss = loss_func(prediction, label.long())
            
            optimier.zero_grad()
            loss.backward()
            optimier.step()
            epoch_loss += loss.detach().item()  # 每一个批次的损失
            
        epoch_loss /= (count + 1)
        epoch_loss_list.append(epoch_loss)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))

        model.eval()
        prediction_true_num = 0
        for iter, (bg, etype, feature, label) in enumerate(dataloader_test):
            bg = bg.to(device)
            etype = etype.to(device)
            feature = feature.to(device)
            label = label.to(device)
            prediction = model(bg, feature, etype)
            prediction = torch.softmax(prediction, 1)
            prediction_class = prediction.argmax(dim=1)
            prediction_true_num += (torch.tensor(prediction_class) == torch.tensor(label)).sum().item()
        epoch_acc = prediction_true_num / len(codeDataSet_test)
        epoch_acc_list.append(epoch_acc)
        print('Epoch {}, prediction_accuracy {:.4f}'.format(epoch, epoch_acc))
    torch.save(model.state_dict(), '../model/java250/model.pth')
    print(epoch_acc_list)
    print(epoch_loss_list)
    with open("./result/epoch_loss_list.txt", "w") as file:

        for item in epoch_loss_list:
            file.write(str(item) + "\n")
    with open("./result/epoch_acc_list.txt", "w") as file:
        for item in epoch_acc_list:
            file.write(str(item) + "\n")
