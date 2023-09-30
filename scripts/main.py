import os
import time
import sys
import torch

# Add parent directory to system path
sys.path.append('..')

from torch.nn import functional as F
from torch.utils.data import DataLoader
from configs.config import Config
from models.model import DeepCoNN
from utils import load_embedding, DeepCoNNDataset, predict_mse, date



def train(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start the training!')
    
    # Compute initial train and validation MSEs
    train_mse = predict_mse(model, train_dataloader, config.device)
    valid_mse = predict_mse(model, valid_dataloader, config.device)
    print(f'{date()}#### Initial train mse {train_mse:.6f}, validation mse {valid_mse:.6f}')
    
    # Start training
    start_time = time.perf_counter()
    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)
    best_loss, best_epoch = 100, 0
    
    for epoch in range(config.train_epochs):
        # Set model to training mode
        model.train()
        total_loss, total_samples = 0, 0
        
        # Iterate over batches in the training data loader
        for batch in train_dataloader:
            user_reviews, item_reviews, ratings = map(lambda x: x.to(config.device), batch)
            predict = model(user_reviews, item_reviews)
            loss = F.mse_loss(predict, ratings, reduction='sum')
            
            # Clear gradients and compute gradients
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_samples += len(predict)

        # Adjust learning rate
        lr_sch.step()
        
        # Set model to evaluation mode
        model.eval()
        
        # Compute train and validation MSEs for current epoch
        valid_mse = predict_mse(model, valid_dataloader, config.device)
        train_loss = total_loss / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; validation mse {valid_mse:.6f}")

        # Save model if it has the best validation MSE so far
        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')


def test(dataloader, model):
    print(f'{date()}## Start the testing!')
    start_time = time.perf_counter()
    
    # Compute test MSE
    test_loss = predict_mse(model, dataloader, config.device)
    
    end_time = time.perf_counter()
    print(f"{date()}## Test end, test mse is {test_loss:.6f}, time used {end_time - start_time:.0f} seconds.")


if __name__ == '__main__':
    config = Config()
    print(config)
    print(f'{date()}  ## Load embedding and data...')
    word_emb, word_dict = load_embedding(config.word2vec_file)

    train_dataset = DeepCoNNDataset(config.train_file, word_dict, config)
    valid_dataset = DeepCoNNDataset(config.valid_file, word_dict, config, retain_rui=False)
    test_dataset = DeepCoNNDataset(config.test_file, word_dict, config, retain_rui=False)
    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size)

    model = DeepCoNN(config, word_emb).to(config.device)
    del train_dataset, valid_dataset, test_dataset, word_emb, word_dict

    os.makedirs(os.path.dirname(config.model_file), exist_ok=True)  
    train(train_dlr, valid_dlr, model, config, config.model_file)
    test(test_dlr, torch.load(config.model_file))

