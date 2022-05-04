from turtle import forward
import numpy as np
import torch
import torch.nn as nn

class PCS_FIR_filter(nn.Module):
    def __init__(self, coeffs_path):
        super(PCS_FIR_filter, self).__init__()

        pcs_coeffs = np.load(coeffs_path)
        num_taps = len(pcs_coeffs)

        self.FIR_filter = nn.Conv1d(1,1,kernel_size=num_taps,padding=num_taps//2,stride=1,bias=False)

        self.FIR_filter.weight.data = torch.FloatTensor(pcs_coeffs).unsqueeze(0).unsqueeze(0)
        for param in self.FIR_filter.parameters():
            param.requires_grad = False

        self.gain = 10
        self.offset = 1.0
    
    def wave_compress(self, x):
        x_sign = torch.sign(x)
        x_abs = torch.abs(x)

        return torch.log(x_abs * self.gain + self.offset) * x_sign
    
    def wave_decompress(self, x):
        x_sign = torch.sign(x)
        x_abs = torch.abs(x)

        return (torch.exp(x_abs)-self.offset) / self.gain * x_sign

    def forward(self, x):
        '''
        Takes x.size() == L, or B*L, or B*1*L
        Returns x.size() == B*L
        '''
        
        if len(list(x.size())) == 1:
            x = x.unsqueeze(0)
        if len(list(x.size())) == 2:
            x = x.unsqueeze(1)
        assert len(list(x.size())) <= 3, 'Pass with dimension: B*1*L, given {}'.format(x.size())
        #x = self.wave_compress(x)
        x = self.FIR_filter(x)
        #x = self.wave_decompress(x)
        return x.squeeze(1)