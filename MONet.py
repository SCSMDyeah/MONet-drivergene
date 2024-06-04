
import random
import sys
from keras.callbacks import EarlyStopping
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import numpy as np

def load_data():
    data = load_hdf_data("...\\CPDB_multiomics.h5", feature_name='features')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names= data
    features = np.concatenate((features[:, 0:16],features[:, 16:32],features[:, 32:48],features[:, 48:64]),axis=1)
    features = torch.FloatTensor(features)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)

    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    num_feats = features.shape[1]
    g = dgl.DGLGraph()
    g.add_nodes(len(features))
    u=[]
    v=[]
    for i in range(adj.shape[0]):
        for j in range(i,adj.shape[0]):
            if adj[i][j]==1.0:
                u.append(i)
                v.append(j)
    src = u+v
    dst = v+u
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    g.ndata['feat'] = features

    y_test = y_test.float()
    y_val = y_val.float()
    y_train = y_train.float()

    g.ndata['y_test'] = y_test
    g.ndata['y_val'] = y_val
    g.ndata['y_train'] = y_train

    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask

    y_label = torch.add(torch.add(y_train,y_val),y_test)
    g.ndata['y_label'] = y_label

    print('Number of nodes:', g.number_of_nodes())
    print('Number of edges:', g.number_of_edges())
    print('Node feature dimensionality:',g.ndata['feat'].shape)

    print('y_test shape:', g.ndata['y_test'].shape)
    print('y_val shape:', g.ndata['y_val'].shape)
    print('y_train shape:', g.ndata['y_train'].shape)

    print('train_mask shape:', g.ndata['train_mask'].shape)
    print('val_mask shape:', g.ndata['val_mask'].shape)
    print('test_mask shape:', g.ndata['test_mask'].shape)

    return g, node_names


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            dglnn.GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
                residual=True,
            )
        )
        self.gat_layers.append(
            dglnn.GATConv(
                hid_size * heads[0],
                out_size,
                heads[1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
                residual=True,
            )
        )

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 1:
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h


class GCN(nn.Module):
    def __init__(self, in_size, hid_size1, hid_size2, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size1, activation=F.relu)
        )
        self.layers.append(dglnn.GraphConv(hid_size1, hid_size2, activation=F.relu))

        self.layers.append(dglnn.GraphConv(hid_size2, out_size))
        self.dropout = nn.Dropout(0.25)

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


class MONet(nn.Module):
    def __init__(self, in_dim, hmonet_dim, out_dim):
        super().__init__()
        heads=[5, 1]
        self.gat = GAT(in_size, 100, out_size, heads).to(device)
        self.gcn = GCN(in_size, 300, 100, out_size).to(device)
        self.fc1 = nn.Linear(in_dim, hmonet_dim)
        self.fc2 = nn.Linear(hmonet_dim, out_dim)
        self.activation = nn.LeakyReLU(0.25)
        self.act = nn.Sigmoid()
        self.dropout = nn.Dropout(0.25)

    def forward(self, g, inputs):
        x_gat = self.gat(g, inputs)
        x_gcn = self.gcn(g, inputs)
        x = torch.cat((x_gat, x_gcn), 1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
def sigmoid(x):
    return 1. / (1 + np.exp(-x))
loss_fcn = nn.BCEWithLogitsLoss(pos_weight=torch.LongTensor([3]).to(device))

def get_prediction(g, features, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = sigmoid(logits.cpu().detach().numpy())
    return logits

def evaluate(g, features, y_val, val_mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[val_mask]
        labels = y_val[val_mask]

        val_loss = loss_fcn(logits, labels)
        logits = sigmoid(logits.cpu().detach().numpy())
        acc = accuracy_score(labels.cpu(), np.round(logits))
        auroc = roc_auc_score(labels.cpu(), logits)
        aupr = average_precision_score(labels.cpu(), logits)

        return logits, val_loss, acc, auroc, aupr

def train(g, features, y_train, y_val, train_mask, val_mask, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    es = EarlyStopping(verbose=False, patience=100)

    for epoch in range(2000):
        model.train()
        logits = model(g, features)

        loss = loss_fcn(logits[train_mask], y_train[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, val_loss, acc, auroc, aupr = evaluate(g, features, y_val, val_mask, model)
        es(val_loss, model)
        if es.early_stop: break
    print(
        "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | auroc {:.4f} | aupr {:.4f} | val_loss {:.4f} ".format(
            epoch, loss.item(), acc ,auroc, aupr, val_loss
        )
    )
    print("epoch used: ", epoch)

if __name__ == "__main__":
    g, node_names = load_data()
    y_train, y_test, y_val = g.ndata['y_train'], g.ndata['y_test'], g.ndata['y_val']
    train_mask, val_mask, test_mask = g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']
    g = g.int().to(device)
    features = g.ndata["feat"].to(device)
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5).to(device)
    norm[torch.isinf(norm)] = 0
    g.ndata["norm"] = norm.unsqueeze(1)
    in_size = features.shape[1]
    out_size = 16
    out_dim = 1
    in_dim = 32

    y_train = y_train.to(device)
    y_test = y_test.to(device)
    y_val = y_val.to(device)

    y_all = np.zeros_like(y_train.cpu().numpy())
    y_all[train_mask] = y_train[train_mask].cpu().numpy()
    y_all[val_mask] = y_val[val_mask].cpu().numpy()
    y_all = torch.FloatTensor(y_all).to(device)
    mask_all = torch.add(train_mask, val_mask)

    masked_values = mask_all.cpu().numpy()
    masked_index = np.where(masked_values)[0]
    masked_y_values = y_all[masked_index].cpu().numpy()
    masked_index = masked_index.reshape((-1, 1))

    print("masked_index.shape: ", masked_index.shape)

    n_train_times = 1
    predictions = []

    hidden = [1024, 32, 64, 128, 256, 512]
    all_logits = []
    for _ in range(len(hidden)):
        random.seed(_)
        np.random.seed(_)
        torch.manual_seed(_)
        model = MONet(in_dim, hidden[_], out_dim).to(device)

        train(g, features, y_train, y_val, train_mask, val_mask, model)
        model.load_state_dict(torch.load("./checkpoint.pt"))
        logits, test_loss, acc, auroc, aupr = evaluate(g, features, y_test, test_mask, model)
        predictions.append(logits)

        all_logits.append(get_prediction(g, features, model))
        old_pred = logits

        print("first model's result")
        print("Test accuracy {:.4f}".format(acc))
        print("Test auroc {:.4f}".format(auroc))
        print("Test aupr {:.4f}".format(aupr))

        del model
    all_predictions = np.squeeze(np.array(predictions))
    all_predictions = all_predictions.mean(axis=0).reshape((-1, 1))

    all_logits = np.squeeze(np.array(all_logits))
    all_logits = all_logits.mean(axis=0).reshape((-1, 1))

    logits = all_predictions
    labels = y_test[test_mask].cpu().numpy()

    acc = accuracy_score(labels, np.round(logits))
    auroc = roc_auc_score(labels, logits)
    aupr = average_precision_score(labels, logits)

    save_predictions(".", node_names, all_logits)
    print("Test accuracy {:.4f}".format(acc))
    print("Test auroc {:.4f}".format(auroc))
    print("Test aupr {:.4f}".format(aupr))

    sys.exit(0)

