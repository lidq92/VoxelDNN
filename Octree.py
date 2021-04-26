import numpy as np


def int2bin(n, count=8): 
    """returns the binary of integer n, using count number of digits""" 
    return [int((n >> y) & 1) for y in range(count-1, -1, -1)]


def deOctree(oct_seq, max_level):  # De octree DFS, by z,y,x
    occupancy_stack = [[[[0, 0, 0]], 0, 1, 0]]
    points = []
    for oct in oct_seq:
        assert len(occupancy_stack) > 0, 'error, maybe max_level too small!'
        fater = occupancy_stack[-1]
        fatertempid, faterboxlen, faterL = fater[1], fater[2], fater[3]
        faterbox = fater[0][fatertempid]
        fatertempid += 1
        occupancy_stack[-1][1] = fatertempid
        if fatertempid == faterboxlen:
            del occupancy_stack[-1]  
        octbin = int2bin(oct)
        boxList = []
        for bit in range(8):
            if octbin[7-bit] == 1:
                bitbin = int2bin(bit, count=3)
                S = 2**(max_level - faterL - 1)
                box = [bitbin[2] * S + faterbox[0], bitbin[1] * S + faterbox[1], bitbin[0] * S + faterbox[2]] #by z,y,x
                boxList.append(box)
        tempIdx, level = 0, faterL+1
        if level != max_level:
            boxLen = len(boxList)
            occupancy_stack.append([boxList, tempIdx, boxLen, level])
        else:
            points.extend(boxList)
    assert len(occupancy_stack) == 0, 'error max_level too large!'
    return points


def combineBoxes2Points(pc_level, departition_level, binstr, boxes):
    boxNum = boxes.shape[0]
    boxCord = deOctree(binstr, departition_level)
    points = []
    for i in range(boxNum):
        box = np.squeeze(boxes[i])
        localPoints = np.vstack(np.where(box==1)).T
        boxpoints = localPoints + np.array(boxCord[i]) * 2 ** (pc_level - departition_level)
        points.extend(boxpoints.tolist())
    return np.array(points)
    
# usage
# ptdecode = combineBoxes2Points(pc_level, departition_level, binstr, decoded_boxes)