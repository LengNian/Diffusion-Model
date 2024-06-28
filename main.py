import os
import time
import torch
import torch.nn as nn
from torchvision import models
from dataset import get_dataloader, get_img_shape
from ddpm import DDPM
# import cv2
import numpy as np
# import eimops
from network import (build_network, convnet_big_cfg, convnet_medium_cfg, convnet_small_cfg, unet_1_cfg, unet_res_cfg)

batch_size = 512
n_epochs = 20


def train(ddpm: DDPM, net, device='cuda', ckpt_path='./model/model.pth'):
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(batch_size)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-3)
    
    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0
    
        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, n_steps, (current_batch_size, )).to(device)
            eps = torch.randn_like(x).to(device)
            
            x_t = ddpm.sample_forward(x, t, eps)
            eps_theta = net(x_t, t.reshape(current_batch_size, 1))
            
            loss = loss_fn(eps_theta, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * current_batch_size

        total_loss /= len(dataloader.dataset)
        toc = time.time()
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')

    # torch.save(net.state_dict(), ckpt_path)
    print('Done.')


configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
    unet_res_cfg
]

if __name__ == '__main__':

    n_steps = 10
    config_id = 4
    device = 'cuda'

    config = configs[config_id]
    net = build_network(config, n_steps)
    ddpm = DDPM(device, n_steps)

    train(ddpm, net, device=device)

    