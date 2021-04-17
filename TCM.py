import math
import torch 
import torch.nn as nn
from spatial_correlation_sampler import SpatialCorrelationSampler
import ipdb


class TCM(nn.Module):   # # Multi-scale Temporal Dynamics Module

    def __init__(self, num_segments, expansion = 1, pos=2):
        super(TCM, self).__init__()
        self.num_segments = num_segments
        self.mtdm = MTDM(num_segments, expansion=expansion, pos=pos)
        self.tam = TAM(num_segments=num_segments, expansion=expansion, pos=pos) 
    
    def forward(self,x):
        out = self.mtdm(x)
        out = self.tam(out)
        return x + out


class TAM(nn.Module):   # Temporal Attention Module
    def __init__(self, num_segments, expansion = 1, pos=2):
        super(TAM, self).__init__()
        self.num_segments = num_segments
        self.expansion = expansion
        self.pos = pos
        self.out_channel = 64*(2**(self.pos-1))*self.expansion
        self.c1 = 16
        self.c2 = 32
        self.c3 = 64

        self.conv1 = nn.Sequential(
        nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1, groups=3, bias=False),
        nn.BatchNorm2d(6),
        nn.ReLU(),
        nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1, groups=3, bias=False),
        nn.BatchNorm2d(6),
        nn.ReLU(),
        nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1, groups=3, bias=False),
        nn.BatchNorm2d(6),
        nn.ReLU(),
        nn.Conv2d(6, self.c1, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(self.c1),
        nn.ReLU()
        )

        self.conv2 = nn.Sequential(
        nn.Conv2d(self.c1, self.c1, kernel_size=3, stride=1, padding=1, groups=self.c1, bias=False),
        nn.BatchNorm2d(self.c1),
        nn.ReLU(),
        nn.Conv2d(self.c1, self.c2, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(self.c2),
        nn.ReLU()
        )
        self.conv3 = nn.Sequential(
        nn.Conv2d(self.c2, self.c2, kernel_size=3, stride=1, padding=1, groups=self.c2, bias=False),
        nn.BatchNorm2d(self.c2),
        nn.ReLU(),
        nn.Conv2d(self.c2, self.c3, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(self.c3),
        nn.ReLU()
        )
        self.conv4 = nn.Sequential(
        nn.Conv2d(self.c3, self.c3, kernel_size=3, stride=1, padding=1, groups=self.c3, bias=False),
        nn.BatchNorm2d(self.c3),
        nn.ReLU(),
        nn.Conv2d(self.c3, self.out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(self.out_channel),
        nn.ReLU()
        )

        k_size = int(math.log(num_segments, 2))
        if ( k_size & 1) == 0:  # is odd number
            k_size = k_size + 1

        self.ETA = eca_layer(self.num_segments, k_size=k_size)   # efficient temporal attention


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # temporal efficient channle attention
        x = x.view((-1, self.num_segments) + x.size()[1:])        # N T C H W
        N,T,C,H,W = x.size()
        x = x.permute(0,2,1,3,4).contiguous()  # N C T H W  
        x = x.view(-1, T, H, W)                # NC,T,H,W
        x = self.ETA(x)
        x = x.view(N,C,T,H,W).permute(0,2,1,3,4).contiguous() # N,T,C,H,W
        x = x.view(-1, C, H, W)
        return x


class MTDM(nn.Module):   # # Multi-scale Temporal Dynamics Module

    def __init__(self, num_segments, expansion = 1, pos=2):
        super(MTDM, self).__init__()
        patchs = [15, 15, 7, 3]
        self.patch = patchs[pos-1]
        self.patch_dilation = 1
        self.soft_argmax = nn.Softmax(dim=1)
        self.expansion = expansion
        self.num_segments = num_segments
        
        self.chnl_reduction = nn.Sequential(
            nn.Conv2d(128*self.expansion, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.matching_layer = Matching_layer(ks=1, patch=self.patch, stride=1, pad=0, patch_dilation=self.patch_dilation)   

    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm) 
    
    def apply_binary_kernel(self, match, h, w, region):
        # binary kernel
        x_line = torch.arange(w, dtype=torch.float).to('cuda').detach()
        y_line = torch.arange(h, dtype=torch.float).to('cuda').detach()
        x_kernel_1 = x_line.view(1,1,1,1,w).expand(1,1,w,h,w).to('cuda').detach()
        y_kernel_1 = y_line.view(1,1,1,h,1).expand(1,h,1,h,w).to('cuda').detach()
        x_kernel_2 = x_line.view(1,1,w,1,1).expand(1,1,w,h,w).to('cuda').detach()
        y_kernel_2 = y_line.view(1,h,1,1,1).expand(1,h,1,h,w).to('cuda').detach()

        ones = torch.ones(1).to('cuda').detach()
        zeros = torch.zeros(1).to('cuda').detach()

        eps = 1e-6
        kx = torch.where(torch.abs(x_kernel_1 - x_kernel_2)<=region, ones, zeros).to('cuda').detach()
        ky = torch.where(torch.abs(y_kernel_1 - y_kernel_2)<=region, ones, zeros).to('cuda').detach()
        kernel = kx * ky + eps
        kernel = kernel.view(1,h*w,h*w).to('cuda').detach()                
        return match* kernel


    def apply_gaussian_kernel(self, corr, h,w,p, sigma=5):
        b, c, s = corr.size()

        x = torch.arange(p, dtype=torch.float).to('cuda').detach()
        y = torch.arange(p, dtype=torch.float).to('cuda').detach()

        idx = corr.max(dim=1)[1] # b x hw    get maximum value along channel
        idx_y = (idx // p).view(b, 1, 1, h, w).float()
        idx_x = (idx % p).view(b, 1, 1, h, w).float()

        x = x.view(1,1,p,1,1).expand(1, 1, p, h, w).to('cuda').detach()
        y = y.view(1,p,1,1,1).expand(1, p, 1, h, w).to('cuda').detach()

        gauss_kernel = torch.exp(-((x-idx_x)**2 + (y-idx_y)**2) / (2 * sigma**2))
        gauss_kernel = gauss_kernel.view(b, p*p, h*w)#.permute(0,2,1).contiguous()

        return gauss_kernel * corr

    def match_to_flow_soft(self, match, k, h,w, temperature=1, mode='softmax'):        
        b, c , s = match.size()     
        idx = torch.arange(h*w, dtype=torch.float32).to('cuda')
        idx_x = idx % w
        idx_x = idx_x.repeat(b,k,1).to('cuda')
        idx_y = torch.floor(idx / w)   
        idx_y = idx_y.repeat(b,k,1).to('cuda')

        soft_idx_x = idx_x[:,:1]
        soft_idx_y = idx_y[:,:1]
        displacement = (self.patch-1)/2
        
        topk_value, topk_idx = torch.topk(match, k, dim=1)    # (B*T-1, k, H*W)
        topk_value = topk_value.view(-1,k,h,w)
        
        match = self.apply_gaussian_kernel(match, h, w, self.patch, sigma=5)
        match = match*temperature
        match_pre = self.soft_argmax(match)
        smax = match_pre           
        smax = smax.view(b,self.patch,self.patch,h,w)
        x_kernel = torch.arange(-displacement*self.patch_dilation, displacement*self.patch_dilation+1, step=self.patch_dilation, dtype=torch.float).to('cuda')
        y_kernel = torch.arange(-displacement*self.patch_dilation, displacement*self.patch_dilation+1, step=self.patch_dilation, dtype=torch.float).to('cuda')
        x_mult = x_kernel.expand(b,self.patch).view(b,self.patch,1,1)
        y_mult = y_kernel.expand(b,self.patch).view(b,self.patch,1,1)
            
        smax_x = smax.sum(dim=1, keepdim=False) #(b,w=k,h,w)
        smax_y = smax.sum(dim=2, keepdim=False) #(b,h=k,h,w)
        flow_x = (smax_x*x_mult).sum(dim=1, keepdim=True).view(-1,1,h*w) # (b,1,h,w)
        flow_y = (smax_y*y_mult).sum(dim=1, keepdim=True).view(-1,1,h*w) # (b,1,h,w)    

        flow_x = (flow_x / (self.patch_dilation * displacement))
        flow_y = (flow_y / (self.patch_dilation * displacement))
            
        return flow_x, flow_y, topk_value     

    def flow_computation(self, x, pos=0, temperature=100):
        
        size = x.size()               
        x = x.view((-1, self.num_segments) + size[1:])        # N T C H W
        x = x.permute(0,2,1,3,4).contiguous() # B C T H W   
                        
        # match to flow            
        k = 1                
        b,c,t,h,w = x.size()            
        t = t-1         

        if pos == 0:
            x_pre = x[:,:,0,:].unsqueeze(dim=2).expand((b,c,t,h,w)).permute(0,2,1,3,4).contiguous().view(-1,c,h,w)
        else:
            x_pre = x[:,:,:-1].permute(0,2,1,3,4).contiguous().view(-1,c,h,w)
            
        #x_pre = x[:,:,0,:].unsqueeze(dim=2).expand((b,c,t-1,h,w))
        x_post = x[:,:,1:].permute(0,2,1,3,4).contiguous().view(-1,c,h,w)

        match = self.matching_layer(x_pre, x_post)    # (B*T-1*group, H*W, H*W)          
        u, v, confidence = self.match_to_flow_soft(match, k, h, w, temperature)
        flow = torch.cat([u,v], dim=1).view(-1, 2*k, h, w)  #  (b, 2, h, w)  
    
        return flow, confidence

    def forward(self,x):
        # multi-scale temporal action feature
        x_redu = self.chnl_reduction(x)
        flow_1, match_v1 = self.flow_computation(x_redu, pos=1)
        flow_2, match_v2 = self.flow_computation(x_redu, pos=0)
        
        x1 = torch.cat([flow_1, match_v1], dim=1)
        x2 = torch.cat([flow_2, match_v2], dim=1)

        _, c, h, w = x1.size()
        x1 = x1.view(-1,self.num_segments-1,c,h,w)
        x2 = x2.view(-1,self.num_segments-1,c,h,w)

        x1 = torch.cat([x1,x1[:,-1:,:,:,:]], dim=1) ## (b,t,3,h,w)
        x2 = torch.cat([x2,x2[:,-1:,:,:,:]], dim=1) ## (b,t,3,h,w)

        out = torch.cat([x1,x2], dim=2)
        out = out.view(-1,2*c,h,w)
        return out

class Matching_layer(nn.Module):
    def __init__(self, ks, patch, stride, pad, patch_dilation):
        super(Matching_layer, self).__init__()
        self.relu = nn.ReLU()
        self.patch = patch
        self.correlation_sampler = SpatialCorrelationSampler(ks, patch, stride, pad, patch_dilation)
        
    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, feature1, feature2):
        feature1 = self.L2normalize(feature1)
        feature2 = self.L2normalize(feature2)
        b, c, h1, w1 = feature1.size()
        b, c, h2, w2 = feature2.size()
        corr = self.correlation_sampler(feature1, feature2)
        corr = corr.view(b, self.patch * self.patch, h1* w1) # Channel : target // Spatial grid : source
        corr = self.relu(corr)
        return corr


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)