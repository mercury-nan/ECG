import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model.CNN import CNN
from model.CNNLSTM import CNNLSTM
from model.WTLSTM import WTLSTM
from model.ResAttnCNNLSTM import EfficientCNNLSTM
import wandb
from utils.data_utils import EcgDataset
import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897" 
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

labels=['N','V','/','L','R']

#hypermeter
model_name = "CNNLSTM"
#model_name = "EfficientCNNLSTM"
epochs = 80
batch_size = 128
learning_rate = 0.0001
step_size = 5
bidirectional = True

#file path
train_path = 'data/train.csv'
val_path = 'data/val.csv'
test_path = 'data/test.csv'

# model = WTLSTM(bidirectional=bidirectional, level=3)
model = CNNLSTM(bidirectional=bidirectional)
#model = EfficientCNNLSTM(bidirectional=bidirectional)

model = model.to(device)

training_data = EcgDataset(file=train_path, labels=labels)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
validation_data = EcgDataset(file=val_path, labels=labels)
val_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
testing_data = EcgDataset(file=test_path, labels=labels)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer, device, batch_size):
    model.train()
    size = len(dataloader.dataset)
    correct = 0
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # count corrects
        correct += (pred.argmax(dim = 1) == y).type(torch.float).sum().item()
        if (batch+1) % step_size == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            correct /= (step_size-1) * batch_size + len(x)
            print(f"Train Accuracy: {(100*correct):>0.1f}%, Train Loss: {loss:>8f}   [{current:>5d}/{size:>5d}]")
            wandb.log({"train_acc": correct, "train_loss": loss})
            correct = .0

def validation_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct = .0, .0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim = 1) == y).type(torch.float).sum().item()
    
    val_loss /= num_batches
    correct /= size
    wandb.log({"val_acc": correct, "val_loss": val_loss})
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return val_loss

def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = .0, .0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim = 1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    wandb.log({"test_acc": correct, "test_loss": test_loss})
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


wandb.init(
    project="ecg-train",
    name=model_name,
    config={
        "learning_rate": learning_rate,
        "architecture": model_name,
        "dataset": train_path,
        "epochs": epochs,
        "batch_size": batch_size,
        "step_size": step_size
    },
    settings=wandb.Settings(init_timeout=300)  # 增加到300秒
    
)


best_val_loss = float('inf')
patience = 5  # 早停的耐心值
no_improve = 0

for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, device, batch_size)
    val_loss = validation_loop(val_dataloader, model, loss_fn, device)
    test_loop(test_dataloader, model, loss_fn, device)
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        # Save the best model
        best_model_path = f"save/{model_name}_best.pth"
        torch.save(model, best_model_path)
    # Uncomment the following lines if you want to implement early stopping
    # else:
    #     no_improve += 1
    #     if no_improve >= patience:
    #         print(f"Early stopping triggered after epoch {epoch+1}")
    #         break

wandb.finish()
print("Done!")

print("Saving final model")
bid_str = "bid" if bidirectional else "uni"
model_path = f"save/{model_name}_{bid_str}_{learning_rate}lr_{epochs}epochs_{batch_size}bs_{step_size}step_size.pth"
torch.save(model, model_path)
