from sched import scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import os
from utils import *
import numpy as np
from model import HydraModel,HydraEvaluator
from options import parse_args
from dataset.dataset import SQLDataset
import datetime
from torch.nn.utils import clip_grad_norm_
if __name__ == "__main__":

    config = parse_args()
    str_ids = config.gpu_ids.split(',')
    config.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
           config.gpu_ids.append(id)
    if len(config.gpu_ids) > 0:
        torch.cuda.set_device(config.gpu_ids[0])
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_ids)
    checkpoint = config.checkpoints_dir
    model_name = config.name
    checkpoint_path = os.path.join(checkpoint, model_name)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    print(torch.cuda.is_available())
    device = torch.device('cuda:{}'.format(config.gpu_ids[0])) if config.gpu_ids else torch.device('cpu')
    print("Device:", device)
    
    #准备数据
    print("Preparing data ...")
    train_path = os.path.join(config.dataroot,config.train_path)
    train_data = SQLDataset(train_path, config, True)
    train_data_loader  = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True)

    num_samples = len(train_data)
    config.num_train_steps = int(num_samples * int(config.epoch) / int(config.batch_size))
    step_per_epoch = num_samples / int(config.batch_size)
    print("total_steps: {0}, warm_up_steps: {1}".format(config.num_train_steps, config.num_warmup_steps))
    
    #初始化模型
    hydramodel = HydraModel(config,device)
    evaluator = HydraEvaluator(checkpoint_path, config,  hydramodel, "")

    #设置optimizer和scheduler
    print("Setting optimizer and scheduler ...")
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in hydramodel.model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": float(config.decay),
        },
        {"params": [p for n, p in hydramodel.model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=float(config.lr))
    schedulers = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.num_warmup_steps),
        num_training_steps=int(config.num_train_steps))
    # 开始训练
    print("start training")
    loss_avg, step = 0.0, 0
    total_epoch = config.epoch
    for epoch in range(total_epoch):
        for batch_id, batch in enumerate(train_data_loader):
            # print(batch_id)
            # cur_loss = model.train_on_batch(batch)
            hydramodel.model.train()
            for k, v in batch.items():
                batch[k] = v.to(device)
            batch_loss = torch.mean(hydramodel.model(**batch)["loss"])
            batch_loss.backward()
            clip_grad_norm_(hydramodel.parameters(), 1.0)
            optimizer.step()
            schedulers.step()
            optimizer.zero_grad()
            cur_loss = batch_loss.cpu().detach().numpy()
            loss_avg = (loss_avg * step + cur_loss) / (step + 1)
            step += 1
            if batch_id % config.print_freq == 0:
                currentDT = datetime.datetime.now()
                print("[{3}] epoch {0}, batch {1}, batch_loss={2:.4f}".format(epoch, batch_id, cur_loss,
                                                                                currentDT.strftime("%m-%d %H:%M:%S")))

        hydramodel.save(checkpoint_path, epoch)
        evaluator.eval(epoch)
