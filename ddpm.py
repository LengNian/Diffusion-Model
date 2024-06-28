import torch

class DDPM():
    def __init__(self, device, n_steps, min_beta=0.0001, max_beta=0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        # product 用于计算alpha的累乘 
        product = 1
        # 为了简便从0开始，就会比论文中少一项
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product

        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    # 前向传播
    def sample_forward(self, x, t, eps=None):
        # self.alpha_bars是一个一维变量
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)

        if eps is None:
            # rand_like生成一个形状与给定张量x相同的随机张量，其元素值来自标准正态分布(均值为0, 标准差为1)
            eps = torch.rand_like(x)
        
        res = torch.sqrt(alpha_bar) * x + eps * torch.sqrt(1 - alpha_bar)

        return res

    # 反向传播
    def sample_backward(self, img_shape, net, device, simple_var=True):
        x = torch.randn(img_shape).to(device)
        net = net.to(device)

        for i in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var)

        return x
    
    # 反向过程中的一个方法
    def sample_backward_step(self, x_t, t, net, simple_var=True):
        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n, dtype=torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                # 分布的方差βt~
                var = (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]
        
        noise = torch.randn_like(x_t)
        noise *= torch.sqrt(var)
        
        # 目标均值μt~
        mean = (x_t - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * eps) / torch.sqrt(self.alphas[t])

        x_t = mean + noise

        return x_t