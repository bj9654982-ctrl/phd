"""Separate uint8 protocol, leaving all uint16 experiments unchanged."""
import cv2,numpy as np
from src.dpdd import DPDDPairs,paired_crop

def read_rgb8(path):
    a=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
    if a is None or a.dtype!=np.uint8:raise ValueError('Expected uint8 RGB')
    return cv2.cvtColor(a,cv2.COLOR_BGR2RGB)

def crop8(x,y,size,seed=None):
    # Exact uint8->uint16 expansion reuses identical paired spatial transforms.
    # k*257/65535 = k/255, without discarding original dataset precision on disk.
    return paired_crop(x.astype(np.uint16)*257,y.astype(np.uint16)*257,size,seed)

class DPDD8Pairs(DPDDPairs):
    def __getitem__(self,key):
        index,seed=key if isinstance(key,tuple) else (key,None)
        s,t=self.pairs[index]
        x,y=crop8(read_rgb8(s),read_rgb8(t),self.patch,seed)
        return {'blur':x,'sharp':y,'source':s.name,'target':t.name}
