import torch
import Unet
import torchvision
import tqdm
def linear_beta_schedule(timesteps=1000):
    return torch.linspace(0.0001, 0.02, timesteps)

def train(
        model,
        data,
        loss_fn,
        optimizer,
        epochs=10,
        device='cuda',
):
    T = 1000
    beta = linear_beta_schedule(T)
    alpha = 1-beta
    alpha_bar = torch.cumprod(alpha, dim=0).to(device=device)
    history = []
    for epoch in range(epochs):
        processBar = tqdm.tqdm(data,unit='step')
        for step, (x_0, _) in enumerate(processBar):
            # 采样的到噪声混合图像
            x_0 = x_0.to(device)
            noise = torch.randn(x_0.shape[0], 1, 28, 28).to(device)
            t = torch.randint(1, T, (x_0.shape[0],)).to(device)
            alpha_bar_t = alpha_bar.gather(dim=0,index=t).to(device)
            x_t = x_0*alpha_bar_t.sqrt().view(-1,1,1,1)+noise*(1-alpha_bar_t).sqrt().view(-1,1,1,1) #channels需要手动匹配
            
            # 去噪
            pre_noise = model(x_t,t)

            # 计算损失并反向传播
            model.zero_grad()
            l = loss_fn(pre_noise, noise)
            l.backward()
            optimizer.step()
            
            if step == len(data)-1:
                history.append(l.item())
            processBar.set_description("[%d/%d] Loss: %.4f" % 
                                   (epoch,epochs,l.item()))
        processBar.close()
if __name__ == '__main__':
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Unet.Unet(
    in_channels=1,
    model_channels=96,
    out_channels=1,
    channel_mult=(1, 2, 2),
    attention_resolutions=[]
    ).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

    path = "./data"
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = torchvision.datasets.MNIST(root=path,train=True,transform=transform,download=True)
    trainDataLoader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
    
    train(net,trainDataLoader,loss_fn,optimizer,epochs=10,device=device)

    # 保存模型权重
    torch.save(net.state_dict(), './model/ddpm_weights.pth')
