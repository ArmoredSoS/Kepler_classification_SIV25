from function import *
from model import *
from utils import *
from torch.utils.data import DataLoader
from math import inf

import torch.nn as nn

def main():
    dataset = create_dataset(200)
    train_ds, test_ds = torch_datasets(dataset, 0.2)
    
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=True)
    
    model = KeplerCNN(activation = nn.LeakyReLU, dropout_rate = 0.5)
    model.apply(init_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    #Scheduler to reduce lr when plateaus are reached, has little impact in final results, probably requires bigger dataset
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    criterion = nn.CrossEntropyLoss()
    n_epochs = 100

    best_metrics = {}
    best_epoch = 0
    best_loss = inf
    patience = 10
    stale = 0

    for epoch in range(n_epochs):
        training_loop(model, train_dl, optimizer, criterion, "cuda", noise=True)
        evaluation, loss = evaluate_model(model, test_dl, criterion, "cuda")
        
        #AI-generated snippet to save the best model
        if loss < best_loss:
            best_loss = loss
            best_metrics = evaluation
            best_epoch = epoch + 1
            #torch.save(model.state_dict(), "best_model.pth")
        else:
            stale += 1
            if stale >= patience:
                break #Not elegant but it works
    
        scheduler.step(loss)
        print(f"Epoch {epoch+1}, Metrics: {evaluation}")
        
    print(f"\nBest performance at epoch {best_epoch}: {best_metrics}")

if __name__ == "__main__":
    main()