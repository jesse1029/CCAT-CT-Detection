import os
import torch

from simclr import SimCLR
from simclr.modules import LARS
from lookahead import Lookahead

def load_optimizer(args, model):

    scheduler = None
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # TODO: LARS
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1, last_epoch=-1)
        print('Use Adam')
    elif args.optimizer =='lookahead':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # TODO: LARS
        optimizer = Lookahead(optimizer, k=5, alpha=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1, last_epoch=-1)
        print('Use lookahead')
    elif args.optimizer =='SGD':
        
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)  # TODO: LARS
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-9, last_epoch=-1)
        print('Use SGD')
        
    elif args.optimizer == "LARS":
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * args.batch_size / 256
        optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0, last_epoch=-1
        )
    else:
        raise NotImplementedError

    return optimizer, scheduler


def save_model(args, model, optimizer, filename=None, best_f1=None, dict_only=False):
    filename = "checkpoint_{}.pth".format(args.current_epoch) if filename is None else filename
    out = os.path.join(args.model_path, filename)
    if isinstance(model, torch.nn.DataParallel):
        model_dict = model.module.state_dict()
    else:
        model_dict = model.state_dict()
    
    if dict_only==False:
        status_dict = {
                'epoch': args.current_epoch,
                'model_state_dict': model_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }
        torch.save(status_dict, out)
    else:
        torch.save(model_dict, out)
        

def load_model(args, model, optimizer=None, useBest=False, filename=None):
    model_fp = os.path.join(args.model_path, "last.pth") if useBest==False else os.path.join(args.model_path, "best.pth")
    if filename is not None:
        model_fp = filename
    checkpoint = torch.load(model_fp)
    print(f'Loading checkpoint{model_fp} and training status!')
    if 'optimizer_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            args.current_epoch = checkpoint['epoch']
        print(f'Current path {model_fp}, where best f1 is { checkpoint["best_f1"]}')
        return checkpoint['best_f1'] if checkpoint['best_f1'] is not None else 0
    else:
        print('Loading checkpoint only!')
        model.load_state_dict(checkpoint)
        return 0
    
    