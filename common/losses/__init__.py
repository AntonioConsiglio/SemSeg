from .ce import CELoss

LOSSES = {
    "ce":CELoss,
    "dice":None,
    "tversky":None
}

def get_loss(loss_cfg:dict):
    
    params = loss_cfg.pop("params",None)
    if params is not None:
        return LOSSES[list(loss_cfg.keys())[0]](**params)

    return LOSSES[list(loss_cfg.keys())[0]]()
