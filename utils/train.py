from tqdm.auto import tqdm
from torch.cuda.amp import autocast
import torch
import matplotlib.pyplot as plt


def train(file_path, model, optimizer, scheduler, criterion, scaler, device, num_epochs, train_loader, val_loader, use_amp=False):
    model.to(device)
    losses_train = []
    losses_val = []
    best_loss = float('Inf')
    best_model = None
    for epoch in tqdm(range(num_epochs)):

        model.train()
        loss_epoch = 0
        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_epoch += loss.item()

        losses_train.append(loss_epoch/len(train_loader))
        scheduler.step()

        model.eval()
        loss_epoch = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_epoch += loss.item()

        losses_val.append(loss_epoch/len(val_loader))

        # progress check
        if epoch % 1 == 0:
            print(
                f'Epoch {epoch}: train loss = {losses_train[-1]}, validation loss = {losses_val[-1]}')

        if losses_val[-1] < best_loss:
            best_loss = losses_val[-1]
            best_model = model.state_dict()

    torch.save(best_model, file_path+'/model.pth')
    plt.figure()
    plt.plot(losses_train, label='Training Loss')
    plt.plot(losses_val, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(file_path+'/loss.png')

    return best_model
