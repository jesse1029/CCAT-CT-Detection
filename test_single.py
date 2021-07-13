import os
import numpy as np
import torch
import torchvision
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from model import save_model, load_model
from utils import yaml_config_hook
from dataset2 import dataset_DFD
from simclr.modules import SingleViT as VViT
import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score
from datetime import datetime
from sklearn import metrics
from setLogger import *
import pandas as pd
def decisionMaker(args, pred):
    
    if 'avg' in args.testMode:
        if isinstance(pred, list):
            pred = torch.vstack(pred)
            pred =torch.mean(pred, 0)
        else:
            pred = pred[0]
        pred =torch.argmax(pred).cpu().numpy()
    elif 'top' in args.testMode:
        k = int(args.testMode.split('-')[-1])
        pred = torch.vstack(pred)
        ratio = pred[:,1]-pred[:,0]
        ratio_amp = torch.sort(torch.abs(ratio), descending=True)
        ind = ratio_amp.indices
        ratio = torch.mean(ratio[ind[:k]])
        pred = 1 if ratio>=0 else 0
    
    return pred

def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    test_dataset = dataset_DFD(0,args, mode = 'test') 

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size if args.test_aug!=1 else 1,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers if args.test_aug!=1 else 0,
    )
    
    model = VViT(0, args, mode='test')

    model = model.to('cuda')
    model.load_state_dict(torch.load(args.model_path))
        
    model.eval()

        
    from tqdm import trange
    if os.path.isdir(args.output_path)==False:
        os.mkdir(args.output_path)
    
      
    fp1, fp2 = open(os.path.join(args.output_path, f'{args.model_path}-covid.csv'), 'w'), open(os.path.join(args.output_path, f'{args.model_path}-non-covid.csv'), 'w')

    with torch.no_grad():
        preds, labs = [], []
        accs, f1,re,pr =[], [],[],[]
        
        outfile = []

        for data in tqdm.tqdm(test_loader):
            if args.test_aug==1:

                if args.evalPerformance:
                    (x,y)=data
                else:
                    x = data
                
                if isinstance(x, list):
                    pred = []

                    for x_it in x:
                        pred_inner=model(x_it, forward_mode='test')
                        pred.extend(pred_inner)
                        filename = x_it['fn']        

                    pred = decisionMaker(args, pred)
                    preds.append(pred)
                else:
                    filename = x['fn']
                    pred=model(x, forward_mode='test')
                    pred = decisionMaker(args, pred)
                    preds.append(pred)
                    
                if args.evalPerformance:
                    labs.extend(y.cpu().numpy())
                
            else:  
                
                if args.evalPerformance:
                    (x,y, issafe)=data
                else:
                    (x, issafe) = data
                
                if issafe:
                    pred = model(x, forward_mode='test')
                    filename = x['fn']
                    if args.evalPerformance:
                        labs.extend(y.cpu().numpy())
                        
                    preds.append(torch.argmax(pred,1).cpu().numpy()[0])
                else:
                    filename = x['fn']
                    pred = model(x, forward_mode='test')
                    if args.evalPerformance:
                        labs.extend(y.cpu().numpy())
                        
                    preds.append(torch.argmax(pred,1).cpu().numpy()[0])
                    LOG.info(f'Found a insufficient CT scan in {x["fn"][0]}')
                    

            if preds[-1]==1:
                fp1.write('%s,' % os.path.basename(filename[0]))
            else:
                fp2.write('%s,' % os.path.basename(filename[0]))
                
            outfile.append([os.path.basename(filename[0]), preds[-1]])
        
        outfile = pd.DataFrame(outfile)
        postfix = 'val' if args.evalPerformance else 'test'
        outfile.to_csv(f"{args.output_path}/{args.model_path}_{postfix}.csv", index=False)
        if args.evalPerformance:   
            
            preds, labs = np.asarray(preds), np.asarray(labs)
            acc = np.mean(preds==labs)
            re = recall_score(labs, preds, average='macro')
            pr = precision_score(labs, preds, average='macro')
            f1 = f1_score(labs, preds, average='macro')
            fpr, tpr, thresholds = metrics.roc_curve(labs, preds, pos_label=1)
            auc = metrics.auc(fpr, tpr)

            
            now = datetime.now() 
            date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
            LOG.info('='*80)
            LOG.info("Test time:"+date_time)
            LOG.info('='*80)
            LOG.info(f'model={args.model_path}, image size={args.image_size}, crop_size={args.crop_size}, max #crops={args.max_det}, test mode={args.testMode}')
            LOG.info(f'(Marco) Validation Accuracy={acc}, Precision={pr}, Recall={re}, F1-Score={f1}, AUC={auc}')
            LOG.info('='*80)

            
        else:
            fp1.close()
            fp2.close()
            LOG.info(f'The number of COVID-19 is {np.sum(np.array(preds)==1)} while the number of normal case is {np.sum(np.array(preds)==0)}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config_single_test.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument('--eval', type=bool, default=True)
    args = parser.parse_args()
    
    
    global LOG
    LOG = init_logging(log_file="%s/%s_test_eval.log" % (args.output_path, args.model_path))
    LOG.info(args)
    print( args.model_path)

    main(args)
