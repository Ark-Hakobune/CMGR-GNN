import torch
import train
import data_loader

# Use existing cfg, but reduce epochs to 5
cfg = train.cfg
print('Original max_epochs:', cfg.max_epochs)
cfg.max_epochs = 50
print('Running training with max_epochs =', cfg.max_epochs)

# Run fit (this may take some time depending on dataset)
encoder, fusion, best_state = train.fit(data_loader.data_train, data_loader.data_test, cfg)
print('Training finished. best_state:', bool(best_state))
