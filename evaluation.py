import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.data_utils import EcgDataset
from pycm import ConfusionMatrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
labels=['N','V','/','L','R']
test_path = 'data/test.csv'
#model_path = 'save/5_CNNbidLSTM.pth'
model_path = 'save/CNNLSTM_best.pth'
model = torch.load(model_path, weights_only=False, map_location=device)
batch_size = 128

testing_data = EcgDataset(file=test_path, labels=labels)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)
loss_fn = nn.CrossEntropyLoss()

def evaluate(dataloader, model, loss_fn, device, labels):
    model.eval()
    size = len(dataloader.dataset)
    test_loss, correct = .0, .0
    matrix = {}
    for i in range(len(labels)):
        matrix[labels[i]]={}
        for j in range(len(labels)):
            matrix[labels[i]][labels[j]] = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            batch_avg_loss = loss_fn(pred, y).item()
            test_loss += batch_avg_loss * x.size(0)
            y_pred = pred.argmax(dim = 1)
            correct += (y_pred == y).type(torch.float).sum().item()
            for label, pred in zip(y, y_pred):
                matrix[labels[label]][labels[pred]] += 1
    cm = ConfusionMatrix(matrix=matrix)
    cm.print_matrix()
    cm.stat(summary=True)
    
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

evaluate(test_dataloader, model, loss_fn, device, labels)
