import torch

def save_checkpoint(model, optimizer, iteration, out):
    to_save = {'model_state':model.state_dict(), 'optimizer_state':optimizer.state_dict(), 'iteration':iteration}
    torch.save(to_save, out)

def load_checkpoint(src, model, optimizer):
    states = torch.load(src)
    model.load_state_dict(states['model_state'])
    optimizer.load_state_dict(states['optimizer_state'])
    return states['iteration']