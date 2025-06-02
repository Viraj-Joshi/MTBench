import numpy as np
import torch

'''
e.g if num_envs = 24576, horizon=32, num_blocks = 6, repeat_idxs = [0,5]
,then orig_len is just the batch size = 24576*32=786432 
each policy genreates orig_len/num_blocks = 131072 samples
when idx is 0, filtered_val keeps all leader data val[0:786432]
when idx is 5, filtered_val keeps all follower data val[786432 + (5-1)*131072:786432 + 5*131072] by finding the correct block

returns entire batch of data (all environments x horizon length) + specific block of data from followers
'''
def filter_leader(val, orig_len, repeat_idxs, num_blocks):
    """
    Filters data corresponding to leader i.e. evaluation policy
    Used with mixed_expl
    """
    if len(val) > 1:
        bsize = orig_len // num_blocks # observations per policy block
        filtered_val = []
        for i, idx in enumerate(repeat_idxs):
            if idx == 0:
                filtered_val.append(val[i*orig_len:(i+1)*orig_len])
            else:
                filtered_val.append(val[i*orig_len + (idx-1)*bsize:i*orig_len + idx*bsize])
        new_val = torch.cat(filtered_val, dim=0)
    else: # axis = 1
        bsize = orig_len // num_blocks
        filtered_val = []
        for i, idx in enumerate(repeat_idxs):
            if idx == 0:
                filtered_val.append(val[:, i*orig_len:(i+1)*orig_len])
            else:
                filtered_val.append(val[:, i*orig_len + (idx-1)*bsize:i*orig_len + idx*bsize])
        new_val = torch.cat(filtered_val, dim=1)
    return new_val

def print_statistics(print_stats, curr_frames, step_time, step_inference_time, total_time, epoch_num, max_epochs, frame, max_frames):
    if print_stats:
        step_time = max(step_time, 1e-9)
        fps_step = curr_frames / step_time
        fps_step_inference = curr_frames / step_inference_time
        fps_total = curr_frames / total_time

        if max_epochs == -1 and max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}')
        elif max_epochs == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}/{max_frames:.0f}')
        elif max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}')
        else:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}/{max_frames:.0f}')