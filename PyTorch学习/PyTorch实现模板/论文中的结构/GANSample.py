import torch as t
import torchvision as tv


# 图片尺寸为3*96*96
class NetG(t.nn.Module):
    def __init__(self, opt):
        # opt为参数配置类
        super().__init__()
        ngf = opt.ngf
        self.main = t.nn.Sequential(t.nn.ConvTranspose2d(opt.nz, ngf*8, 4, 1, 0, bias=False),
                                    t.nn.BatchNorm2d(ngf*8),
                                    t.nn.ReLU(True),
                                    t.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                                    t.nn.BatchNorm2d(ngf * 4),
                                    t.nn.ReLU(True),
                                    t.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                                    t.nn.BatchNorm2d(ngf * 2),
                                    t.nn.ReLU(True),
                                    t.nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
                                    t.nn.BatchNorm2d(ngf * 1),
                                    t.nn.ReLU(True),
                                    t.nn.ConvTranspose2d(ngf * 1, 3, 5, 3, 1, bias=False),
                                    t.nn.Tanh()
                                    )

    def forward(self, input):
        # input尺寸为opt.nz*1*1
        return self.main(input)


class NetD(t.nn.Module):
    def __init__(self, opt):
        super().__init__()
        ndf = opt.ndf
        self.main = t.nn.Sequential(t.nn.Conv2d(3, ndf * 1, 5, 3, 1, bias=False),
                                    t.nn.LeakyReLU(0.2, inplace=True),
                                    t.nn.Conv2d(ndf * 1, ndf * 2, 4, 2, 1, bias=False),
                                    t.nn.BatchNorm2d(ndf * 2),
                                    t.nn.LeakyReLU(0.2, inplace=True),
                                    t.nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                                    t.nn.BatchNorm2d(ndf * 4),
                                    t.nn.LeakyReLU(0.2, inplace=True),
                                    t.nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                                    t.nn.BatchNorm2d(ndf * 8),
                                    t.nn.LeakyReLU(0.2, inplace=True),
                                    t.nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                                    t.nn.Sigmoid()
                                    )

    def forward(self, input):
        return self.main(input).view(-1)


class Config(object):
    data_path = 'data\\'
    num_workers = 4  # 加载数据的进程数
    image_size = 96
    batch_size = 256
    max_epoch = 200
    lrg = 2e-4  # G的学习速率
    lrd = 2e-4  # D的学习速率
    beta1 = 0.5  # Adam的优化参数
    use_gpu = True
    nz = 100  # 噪声维数
    ngf = 64
    ndf = 64

    save_path = 'imgs\\'
    debug_file = 'tmp\\debuggan'
    d_every = 1  # 每1个batch训练一次D
    g_every = 5  # 每5个batch训练一次G
    decay_every = 10
    netd_path = 'checkpoints\\netd.pth'
    netg_path = 'checkpoints\\netg.pth'

    gen_img = 'result.png'
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0
    gen_std = 1


if __name__ == '__main__':
    opt = Config()
    # 数据加载与预处理
    transforms = tv.transforms.Compose([tv.transforms.Scale(opt.image_size),
                                        tv.transforms.CenterCrop(opt.image_size),
                                        tv.transforms.ToTensor(),
                                        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    dataloader = t.utils.data.DataLoader(dataset, batch_size = opt.batch_size, shuffle=True, num_workers=opt.num_workers, drop_last=True)
    # 定义模型
    netd = NetD(opt)
    netg = NetG(opt)
    # 定义优化器和损失函数
    optimizer_g = t.optim.Adam(netg.parameters(), opt.lrg, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), opt.lrd, betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss()
    # 训练标签和噪音
    true_labels = t.ones(opt.batch_size)
    fake_labels = t.zeros(opt.batch_size)
    fix_noises = t.randn(opt.batch_size, opt.nz, 1, 1)
    noises = t.randn(opt.batch_size, opt.nz, 1, 1)
    # GPU化
    if opt.use_gpu:
        netd.cuda()
        netg.cuda()
        criterion.cuda()
        true_labels, fake_labels = true_labels.cuda(), fake_labels.cuda()
        fix_noises, noises = fix_noises.cuda(), noises.cuda()

    # 开始训练
    for ii, (img, _) in enumerate(dataloader):
        # 读取图像数据
        real_img = img
        if opt.use_gpu:
            real_img = real_img.cuda()
        # 训练判别器
        if (ii+1) % opt.d_every == 0:
            # 初始化梯度为零
            optimizer_d.zero_grad()
            # 真图片训练
            output = netd(real_img)
            error_d_real = criterion(output, true_labels)
            error_d_real.backward()
            # 假图片训练
            noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
            fake_img = netg(noises).detach()  # 阶段梯度传递，用detach()的方法
            fake_output = netd(fake_img)
            error_d_fake = criterion(fake_output, fake_labels)
            error_d_fake.backward()
            # 更新参数
            optimizer_d.step()
        # 训练生成器
        if (ii+1) % opt.g_every == 0:
            # 初始化梯度为零
            optimizer_g.zero_grad()
            noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
            fake_img = netg(noises)
            fake_output = netd(fake_img)
            error_g = criterion(fake_output, true_labels)
            error_g.backward()
            # 更新参数
            optimizer_g.step()

    # 训练好，新生成样本
    netg.eval()
    netd.eval()
    noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
    fake_img = netg(noises)
    scores = netd(fake_img)