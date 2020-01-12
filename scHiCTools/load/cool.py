# -*- coding: utf-8 -*-
import cooler

def read_cool(file, chrom, resolution, resolution_adjust):
    
    c = cooler.Cooler(file)
    f=c.matrix(balance=False, as_pixels=True, join=True ).fetch(chrom)
    f=f[:,[0,1,3,4,6]]
    
    for lst in f:
                
        # pos1 pos2
        if resolution_adjust:
            p1, p2 = int(lst[1]) // resolution, int(lst[3]) // resolution
        else:
            p1, p2 = int(lst[1]), int(lst[3])
            
            

        yield p1, p2, lst[4]

