import argparse
import numpy as np
from voxelDNN_Inference import occupancy_map_explore, set_global_determinism
from voxelDNN_meta_endec import load_compressed_file
import gzip
import pickle
import arithmetic_coding
from voxelDNN import VoxelDNN
import tensorflow as tf
import time

# encoding from breadth first sequence for parallel computing
def voxelDNN_decoding(args):
    ply_path, model_path, outputfile, metadata, flags_pkl = args
    start = time.time()
    # reading metadata
    with gzip.open(metadata, "rb") as f:
         decoded_binstr, pc_level, departition_level = load_compressed_file(f)
    # getting encoding input data
    boxes, binstr, no_oc_voxels = occupancy_map_explore(ply_path, pc_level, departition_level)

    with open(flags_pkl,'rb') as f:
        flags = pickle.load(f) # flags are contained in a pkl file, collecting information purpose

    bbox_max = 2 ** (pc_level - departition_level)
    voxelDNN = VoxelDNN(bbox_max, bbox_max, bbox_max)
    voxel_DNN = voxelDNN.restore_voxelDNN(model_path)
    with open(outputfile, "rb") as inp:
        bitin = arithmetic_coding.BitInputStream(inp)
        dec = arithmetic_coding.ArithmeticDecoder(32, bitin)
        decoded_boxes = np.zeros_like(boxes)
        for i in range(len(boxes)):
#         for i in range(1):
            decoded_boxes[i] = decompress_from_adaptive_freqs(decoded_boxes[i], np.asarray(boxes[i]), flags[i][1], dec, voxel_DNN, bbox_max)
    decoded_boxes = decoded_boxes.astype(int)
    end = time.time()
    print('Encoding time: ', end - start)
    
    boxes = boxes.astype(int)
    compare = decoded_boxes == boxes
    print('Check 1: decoded pc level: ',pc_level)
    print('Check 2: decoded block level',  departition_level)
    print('Check 3: decoded binstr ', binstr == decoded_binstr)
    print('Check 4: decoded boxes', np.count_nonzero(compare)/np.prod(compare.shape), np.count_nonzero(compare), '/', np.prod(compare.shape), compare.all())
#     err = decoded_boxes - boxes
#     print(err.max(), err.min())
#     print(decoded_boxes.max(), decoded_boxes.min())


def decompress_from_adaptive_freqs(decoded_box, box, flags, dec, voxel_DNN, bbox_max, start=[0,0,0]):
    box_size = box.shape[0]
    print('Number of non empty voxels: ', np.sum(box))
    fl_idx = 0
#     print('Flags: ', flags[0:9])
    if flags[fl_idx] == 1:
        print('Decoding as a single box')
        # causality is preserved, using original box to decode is ok
        # (lossless compression + mask filters -> OK)
        fake_box = np.zeros((1, bbox_max, bbox_max, bbox_max, 1))
        fake_box[:, 0:box_size, 0:box_size, 0:box_size, :] = box
        probs = tf.nn.softmax(voxel_DNN(fake_box)[0, :, :, :, :], axis=-1)
        probs = probs[0:box_size, 0:box_size, 0:box_size, :]
        probs = np.asarray(probs, dtype='float32')
        print(probs.shape)
        for d in range(box_size):
            for h in range(box_size):
                for w in range(box_size):
                    fre = [probs[d, h, w, 0], probs[d, h, w, 1], 0.]
                    fre = np.asarray(fre)
                    fre = (2 ** 10 * fre)
                    fre = fre.astype(int)
                    fre += 1
                    freq = arithmetic_coding.NeuralFrequencyTable(fre)
                    symbol = dec.read(freq)
                    decoded_box[start[0] + d, start[1] + h, start[2] + w, 0] = symbol
#         symbol = dec.read(freq) 
        del flags[fl_idx]
    elif flags[fl_idx] == 0:
        del flags[fl_idx]
    else:
        del flags[fl_idx]
        child_bbox_max = int(box_size / 2)
        for d in range(2):
            for h in range(2):
                for w in range(2):
                    child_box = box[d * child_bbox_max:(d + 1) * child_bbox_max,
                                    h * child_bbox_max:(h + 1) * child_bbox_max,
                                    w * child_bbox_max:(w + 1) * child_bbox_max, :]
                    new_start = [start[0] + d * child_bbox_max, start[1] + h * child_bbox_max, start[2] + w * child_bbox_max]
                    decoded_box = decompress_from_adaptive_freqs(decoded_box, child_box, flags, dec, voxel_DNN, bbox_max, new_start)
                                                            
    return decoded_box


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("-ply", '--plypath', type=str, 
                        help='path to input ply file')
    parser.add_argument("-model", '--modelpath', type=str, 
                        help='path to input model .h5 file')
    parser.add_argument("-output", '--outputfile', type=str,
                        help='name of output file')
    parser.add_argument("-metadata", '--output_metadata', type=str,
                        help='name of output file')
    parser.add_argument("-heatmap", '--output_heatmap', type=str,
                        help='name of output heatmap pkl')
    args = parser.parse_args()
    
    set_global_determinism(seed=42)
    
    voxelDNN_decoding([args.plypath, args.modelpath, args.outputfile, args.output_metadata, args.output_heatmap])
