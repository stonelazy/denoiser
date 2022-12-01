import math
import time

import torch
from torch import nn
from torch.nn import functional as F

import math
import time

import torch
from torch import nn
from torch.nn import functional as F

def diff(ta, tb):
       print(f"delta batch/streaming: {torch.norm(ta - tb) / torch.norm(ta):.2%}")

class Umucs:
    def __init__(self) -> None:            
        self.depth=3
        self.stride_inp=4
        
        self.c1 = nn.Conv1d(in_channels=1, out_channels=48, kernel_size=8, stride=4)
        self.c2 = nn.Conv1d(in_channels=48, out_channels=96, kernel_size=8, stride=4)
        self.c3 = nn.Conv1d(in_channels=96, out_channels=192, kernel_size=8, stride=4)
        self.encoder=nn.ModuleList()
        
        self.encoder.append(self.c1)
        self.encoder.append(self.c2)
        self.encoder.append(self.c3)
        
        # self.inp1= torch.randn(1,296)
        
        self.stride = self.stride_inp ** self.depth
        self.frame_length= self.valid_length(1)

    def get_out(self, length,depth):
        kernel_size=8
        stride=4
        resample=1
        length = math.ceil(length * resample)
        for idx in range(depth):
            length = math.ceil((length - kernel_size) / stride) + 1
            length = max(length, 1)
        return length

    def get_in(self,length,depth):
        """
        Determine the input_length given that we got `length` in the output.
        """
        kernel_size=8
        stride=4
        resample=1
        length = math.ceil(length * resample)
        for idx in range(depth):
            length = (length - 1) * stride + kernel_size
        length = int(math.ceil(length / resample))
        return int(length)

    def valid_length(self,length):
        len = self.get_out(length,self.depth)
        return self.get_in(len,self.depth)
    
    def frm_zmucs(self,inp):
        self.pending = torch.zeros(1, 0)
        self.pending=torch.cat([self.pending,inp],dim=1)
        inp_frames = []
        while self.pending.shape[-1] >= self.frame_length:
            frame = self.pending[:,:self.frame_length]
            inp_frames.append(frame)
            self.pending = self.pending[:,self.stride:] 
        
        return inp_frames           
        
        
    def feed(self,inp:torch.Tensor,conv_state):
        
        expected_length = self.valid_length(1)
        assert inp.shape[-1] == expected_length
        
        # do_predict(inp)
        
        return_values = [
            inp[:,self.stride:],
        ]
        return return_values

    def framed_inp(self,inp):
        return inp.unfold(1,self.valid_length(1),self.stride).squeeze(0)
    
    def main(self,inp):
        framed_inp = self.framed_inp(inp)
        # zms_fr = torch.stack(self.frm_zmucs(inp)).transpose(0,1)
        # assert torch.allclose(zms_fr,framed_inp) 
        
        self.conv_state = []
        
        out = []
        for each in framed_inp:
            out.append(self.pred_frame(each.unsqueeze(0)))
        
        output = torch.stack(out).transpose(0,3).squeeze(0)
        return output
    
    def pred_frame(self,frame):
        x = frame[None]
        
        first = len(self.conv_state) == 0
                
        next_state = []
        for idx, encode in enumerate(self.encoder):
            
            stride = self.stride // (self.stride_inp ** idx)
            
            if not first:
                want = self.get_in(stride,1) 
                x = x[..., -want:]
            
            x = encode(x)
                 
            if not first:
                prev = self.conv_state.pop(0)
                x = torch.cat([prev, x], -1)

            next_state.append(x[..., stride:])
        
        self.conv_state = next_state
        return x
            
if __name__ == "__main__":
    umucs = Umucs()
    inp = torch.randn(1,1600)
    # umucs.frm_zmucs(inp)
    online_op = umucs.main(inp)
    offl_op = umucs.c3(umucs.c2(umucs.c1(inp[None])))
    diff(offl_op,online_op)
    assert torch.allclose(offl_op,online_op,1e-6,1e-6)
    print("exit over")