import torch
from nasflow.optim.rmsprop_tf import RMSpropTF

def get_optimizer_with_lr(model, lr, optimizer='sgd'):
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr,
                                    momentum=0.9,
                                    weight_decay=0,
                                    nesterov=False)
    elif optimizer == 'sgd++':
        optimizer = torch.optim.SGD(model.parameters(), lr,
                                    momentum=0.9,
                                    weight_decay=0,
                                    nesterov=True)
    elif optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr, eps=0.01)
    elif optimizer == 'rmsprop-tf':
        # This is in-consistent with TF-Rmsprop. eps should be tuned.'
        print("Using RMSProp-TF optimizer.")
        optimizer = RMSpropTF(model.parameters(), lr, eps=1e-3, momentum=.9)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr, eps=1e-4, weight_decay=0)
    else:
        raise NotImplementedError

    return optimizer
