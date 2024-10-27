import torch
import numpy as np
import gc
import os
import nibabel as nib
import random
import argparse
from numpy.typing import NDArray
import torch.nn.functional as F
import sys
from models.MedSegMamba import MedSegMamba
from nibabel.processing import conform as nib_conform
from conform import rescale, conform, is_conform #fastsurfer conform functions
print('imports done')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default = 1000, type=int)

parser.add_argument('--input_path', default='', type=str) 
parser.add_argument('--output_T1_path', default=None, type=str) 
parser.add_argument('--output_aseg_path', default=None, type=str)
parser.add_argument('--output_hsf_path', default = None, type=str)

parser.add_argument('--no_conform', default = False, type=bool)

parser.add_argument('--aseg_model_path', default='', type=str)
parser.add_argument('--hsf_model_path', default='', type=str)

parser.add_argument('--gpu_id', default=0, type=int) 

parser.add_argument('--aseg_step_size', default=32, type=int)
args = parser.parse_args()

def pad_or_crop_array(array, padding):
    slices = []
    for i, (pad_before, pad_after) in enumerate(padding):
        shape = array.shape[i]
        start = max(0, -pad_before)
        end = shape - max(0, -pad_after)
        slices+=[start, end]
    
    cropped_array = array[slices[0]:slices[1],slices[2]:slices[3],slices[4]:slices[5]]
    
    final_padding = []
    for pad_before, pad_after in padding:
        pad_before = max(0, pad_before)
        pad_after = max(0, pad_after)
        final_padding.append((pad_before, pad_after))
    
    padded_array = np.pad(cropped_array, final_padding, mode='constant', constant_values=0)
    return padded_array

def sequential_patch_iter(image: NDArray, patch_size=96, step=32):
    (H, W, D) = image.shape
    count=0
    #image_zeropadding = patch/2
    for z in range(0, D - patch_size+1, step):
        for y in range(0, W- patch_size+1, step):
            for x in range(0, H - patch_size+1, step):

                patch = np.float32(image[x : x + patch_size, y : y + patch_size, z : z + patch_size])

                coordinate = (x, y, z)
                count=count+1
                
                yield patch.squeeze(), coordinate, count #count starts from 1

def run_and_reconstruct(model, num_classes, image: NDArray, patch_size=96, step=32, device = torch.device('cuda')):
    
    (H, W, D) = image.shape
    reconstructed = torch.zeros((num_classes, H, W, D))
    
    model.eval()
    for patch, coordinate, count in iter(sequential_patch_iter(image, patch_size, step)):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del count
        (x, y, z) = coordinate
        patch = torch.from_numpy(patch).float().squeeze().unsqueeze(0).unsqueeze(0)
        patch = patch.to(device)
        predicted_patch = None
        with torch.no_grad():
            predicted_patch = model(patch).cpu() # 1 1 96 96 96
            del patch
        #add the voted probability
        reconstructed[:, x : x + patch_size, y : y + patch_size, z : z + patch_size] = reconstructed[:, x : x + patch_size, y : y + patch_size, z : z + patch_size] + predicted_patch.squeeze()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reconstructed = reconstructed.squeeze()
    reconstructed = F.softmax(reconstructed, dim=0)
    return reconstructed.numpy()

def before_and_after(array):
    before = np.argmax(array)
    after = len(array)-np.sum(array)-before
    return before, after

def crop_images_full_flush_single(conformed_img , step_size): #158, 170, 202 max raw

    valid_slices_x = ~np.all(conformed_img == 0, axis=(1, 2))
    valid_slices_y = ~np.all(conformed_img == 0, axis=(0, 2))
    valid_slices_z = ~np.all(conformed_img == 0, axis=(0, 1))

    pad_before_x, pad_after_x = before_and_after(valid_slices_x)
    pad_before_y, pad_after_y = before_and_after(valid_slices_y)
    pad_before_z, pad_after_z = before_and_after(valid_slices_z)

    # Crop the array to remove zero slices along each dimension
    cropped_conformed_img = conformed_img[valid_slices_x][:, valid_slices_y][:, :, valid_slices_z]
    
    original_shape = cropped_conformed_img.shape
    #print(original_shape)
    (depth_pad, height_pad, width_pad) = tuple((step_size-((dim-96) % step_size)) for dim in original_shape)
    
    depth_pad_before = depth_pad // 2
    depth_pad_after = depth_pad - depth_pad_before
    
    pad_before_x -= depth_pad_before
    pad_after_x -= depth_pad_after

    height_pad_before = height_pad // 2
    height_pad_after = height_pad - height_pad_before

    pad_before_y -= height_pad_before
    pad_after_y -= height_pad_after

    width_pad_before = width_pad // 2
    width_pad_after = width_pad - width_pad_before

    pad_before_z -= width_pad_before
    pad_after_z -= width_pad_after
    
    final_conformed_img = pad_or_crop_array(cropped_conformed_img, ((depth_pad_before, depth_pad_after),
                                                           (height_pad_before, height_pad_after),
                                                           (width_pad_before, width_pad_after)))
    
    padding = ((pad_before_x, pad_after_x),(pad_before_y, pad_after_y),(pad_before_z, pad_after_z))
    #return conformed_img, ((0,0),(0,0),(0,0))
    return final_conformed_img, padding

def map_fs_aseg_labels(volume, orig_mapping_dict = {0: 0, 4: 1, 5: 2, 7: 3, 8: 4, 10: 5, 11: 6, 12: 7, 13: 8, 14: 9, 15: 10, 16: 11, 17: 12, 18: 13, 24: 14, 26: 15, 28: 16, 31: 17, 43: 18, 44: 19, 46: 20, 47: 21, 49: 22, 50: 23, 51: 24, 52: 25, 53: 26, 54: 27, 58: 28, 60: 29, 63: 30, 77: 31}):
    #valid_classes = [0, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63, 77]
    #mapping = {value: index for index, value in enumerate(valid_classes)}
    reversed_mapping_dict = {value: key for key, value in orig_mapping_dict.items()}
    mapped_volume = np.vectorize(reversed_mapping_dict.get)(volume)
    return mapped_volume
def map_fs_hsf_labels(volume, FS60_hipp_classes = [0, 1203, 1204, 1205, 1206, 1208, 1209, 1210, 1211, 1212, 1214, 1215, 1226,
                        2203, 2204, 2205, 2206, 2208, 2209, 2210, 2211, 2212, 2214, 2215, 2226]):
    mapping = {index : value for index, value in enumerate(FS60_hipp_classes)}
    mapped_volume = np.vectorize(mapping.get)(volume)
    return mapped_volume

def run_aseg(aseg_model, input_img_path, no_conform = True, aseg_save_path = None, conformed_input_save_path = None, step_size = 32, device = torch.device('cuda')):
    print(f'step size {step_size}')
    fs_affine_matrix = np.array([
        [-1.0, 0.0, 0.0, 127.0],
        [0.0, 0.0, 1.0, -145.0],
        [0.0, -1.0, 0.0, 147.0],
        [0.0, 0.0, 0.0, 1.0]])
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # print()
    if not os.path.exists(input_img_path):
        return
    
    input_img = nib.load(input_img_path)
    orig_shape = input_img.shape
    orig_header = input_img.header
    orig_affine = input_img.affine
    conformed_input_img = None
    if no_conform:
        print('input shape', orig_shape)
        print(nib.aff2axcodes(input_img.affine))
        if nib.aff2axcodes(input_img.affine) ==('R', 'A', 'S'):
            print('rotate to LIA', flush=True)
            new_shape = (orig_shape[0], orig_shape[2], orig_shape[1])
        else:
            new_shape = (256,256,256) #default

        input_img = nib_conform(input_img, new_shape, orientation = 'LIA')
    else:
        if not is_conform(input_img, verbose = False):
            print('conform', input_img_path, flush=True)
            input_img = conform(input_img)
    if conformed_input_save_path is not None:
        try:
            nib.save(input_img, conformed_input_save_path)
            print('saved conformed t1:', conformed_input_save_path, flush=True)
        except:
            print('invalid conformed t1w dst:', conformed_input_save_path, flush=True)
    else:
        print('dont save conformed input', flush=True)

    input_img = input_img.get_fdata()
    scaled_raw_conformed_img = rescale(input_img, 0, 255)
    scaled_raw_conformed_img[input_img==0]==0
    conformed_input_img = np.uint8(np.rint(scaled_raw_conformed_img)) / 255 # convert to uchar like in conform and then scale between 0 and 1
    
    print(np.min(conformed_input_img))
    print(np.max(conformed_input_img))
    print('conformed and rescaled intensity', flush=True)
    del scaled_raw_conformed_img

    #ipdb.set_trace()
    cropped_conformed_img, padding = crop_images_full_flush_single(conformed_input_img, step_size)
    #ipdb.set_trace()
    print(cropped_conformed_img.shape, flush=True)
    print(padding, flush=True)
    print('aseg inference', flush=True)
    full_predicted_scan = run_and_reconstruct(aseg_model, num_classes = 32, image = cropped_conformed_img, patch_size=96, step=step_size, device = device)
    full_predicted_scan = np.argmax(full_predicted_scan, axis = 0)
    
    padded_full_predicted_scan = pad_or_crop_array(full_predicted_scan, padding)
    print(padded_full_predicted_scan.shape, flush=True)
    #print(np.unique(padded_full_predicted_scan), flush=True)
    padded_full_predicted_scan = map_fs_aseg_labels(padded_full_predicted_scan)
    # print(np.unique(padded_full_predicted_scan), flush=True)
    aseg = nib.Nifti1Image(np.uint16(padded_full_predicted_scan), affine=fs_affine_matrix, header = orig_header)
    aseg.header.set_data_dtype(np.uint16)
    if aseg_save_path is not None:
        if no_conform:
            save_aseg = nib_conform(aseg, orig_shape, orientation = 'RAS', order=0)
            print(nib.aff2axcodes(save_aseg.affine))
            print('aseg shape', save_aseg.get_fdata().shape)
            save_aseg = nib.Nifti1Image(np.uint16(save_aseg.get_fdata()), affine=orig_affine, header=orig_header)
            save_aseg.header.set_data_dtype(np.uint16)
            print(np.unique(save_aseg.get_fdata()), len(np.unique(save_aseg.get_fdata())), flush=True)
        else:
            save_aseg = aseg
        try:
            nib.save(save_aseg, aseg_save_path)
            print('saved aseg:', aseg_save_path, flush=True)
        except:
            print('invalid aseg dst:', aseg_save_path, flush=True)
    else:
        print("dont save aseg", flush=True)
    print('aseg done', flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return aseg, conformed_input_img, orig_header, orig_affine, orig_shape
def get_cropped_input(aseg, input_img):
    aseg_shape = aseg.shape
    valid_aseg_classes = [17, 53]
    input_aseg_masked = np.isin(aseg, valid_aseg_classes)

    #get bounding box 96 x 96 x 96
    valid_slices_x = ~np.all(input_aseg_masked == 0, axis=(1, 2)).squeeze()
    valid_slices_y = ~np.all(input_aseg_masked == 0, axis=(0, 2)).squeeze()
    valid_slices_z = ~np.all(input_aseg_masked == 0, axis=(0, 1)).squeeze()

    first_x = np.where(valid_slices_x == 1)[0][0]
    last_x = np.where(valid_slices_x == 1)[0][-1]

    first_y = np.where(valid_slices_y == 1)[0][0]
    last_y = np.where(valid_slices_y == 1)[0][-1]

    first_z = np.where(valid_slices_z == 1)[0][0]
    last_z = np.where(valid_slices_z == 1)[0][-1]


    x_pad = 96 - (last_x - first_x + 1)
    x_pad_before = x_pad//2
    x_pad_after = x_pad - x_pad_before

    valid_slices_x[first_x - x_pad_before : last_x + x_pad_after + 1] = 1

    y_pad = 96 - (last_y - first_y + 1)
    y_pad_before = y_pad//2
    y_pad_after = y_pad - y_pad_before
    
    valid_slices_y[first_y - y_pad_before : last_y + y_pad_after + 1] = 1

    z_pad = 96 - (last_z - first_z + 1)
    z_pad_before = z_pad//2
    z_pad_after = z_pad - z_pad_before

    valid_slices_z[first_z - z_pad_before : last_z + z_pad_after + 1] = 1

    cropped_img = input_img[valid_slices_x][:, valid_slices_y][:, :, valid_slices_z]
    
    padding_x = (first_x - x_pad_before, aseg_shape[0] - last_x - x_pad_after-1)
    padding_y = (first_y - y_pad_before, aseg_shape[1] - last_y - y_pad_after-1)
    padding_z = (first_z - z_pad_before, aseg_shape[2] - last_z - z_pad_after-1)
    padding = (padding_x, padding_y, padding_z)
    print(cropped_img.shape, flush=True)
    return torch.from_numpy(cropped_img.squeeze()).unsqueeze(0).unsqueeze(0), padding
def run_hsf(hsf_model, aseg, input_img, orig_shape = None, no_conform = False, orig_affine = None, hsf_save_path = '', orig_header = None, device = torch.device('cuda')):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    fs_affine_matrix = np.array([
        [-1.0, 0.0, 0.0, 127.0],
        [0.0, 0.0, 1.0, -145.0],
        [0.0, -1.0, 0.0, 147.0],
        [0.0, 0.0, 0.0, 1.0]])
    cropped_input, padding = get_cropped_input(aseg, input_img)
    
    cropped_input = cropped_input.float().to(device)
    predicted_patch = None
    hsf_model.eval()
    with torch.no_grad():
        predicted_patch = hsf_model(cropped_input).cpu()
        
    predicted_patch = np.argmax(np.array(predicted_patch.squeeze()), axis = 0).squeeze()
    
    full_predicted_scan = pad_or_crop_array(predicted_patch, padding)
    full_predicted_scan = map_fs_hsf_labels(full_predicted_scan)
    
    seg = nib.Nifti1Image(np.uint16(full_predicted_scan), affine=fs_affine_matrix, header = orig_header)
    seg.header.set_data_dtype(np.uint16)
    
    if no_conform:
        save_seg = nib_conform(seg, orig_shape, orientation = 'RAS', order = 0)
        print(nib.aff2axcodes(save_seg.affine))
        print('hsf shape', save_seg.get_fdata().shape)
        save_seg = nib.Nifti1Image(np.uint16(save_seg.get_fdata()), affine=orig_affine, header=orig_header)
        save_seg.header.set_data_dtype(np.uint16)
        print(np.unique(save_seg.get_fdata()), len(np.unique(save_seg.get_fdata())), flush=True)
    else:
        save_seg = seg
    try:
        nib.save(save_seg, hsf_save_path)
        print('saved hsf:', hsf_save_path, flush=True)
    except:
        print('invalid hsf dst:', hsf_save_path, flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return seg


if __name__ == '__main__':

    if ((args.output_aseg_path is None) and (args.output_hsf_path is None)):
        raise Exception("must provide a path to save the aseg or hsf.")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    print(os.getpid())
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if (args.gpu_id != None) and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, flush=True)
    
    aseg_model = MedSegMamba(img_dim = 96,
        patch_dim = 8,
        img_ch = 1,
        output_ch = 32,
        channel_sizes = [1,32,64,128,256,1024], 
        mamba_d_state = 64, # for vss block
        num_layers = 9, #6 or 9
        vss_version = 'v5', # None for vanilla
        mlp_dropout_rate = 0, #0.1
        attn_dropout_rate = 0, #0.1
        drop_path_rate=0, #0
        ssm_expansion_factor=1,
        id = False, #TABSurfer default
        preact = False, #TABSurfer default
        maxpool = True, #TABSurfer default
        upsample = True, #TABSurfer default
        full_final_block=True,
        scan_type='scan'
        )
    aseg_model.load_state_dict(torch.load(args.aseg_model_path, map_location=torch.device('cpu')))
    aseg_model.to(device)
    print('run aseg', args.aseg_model_path)
    aseg, input_img, header, affine, shape = run_aseg(aseg_model, args.input_path, no_conform=args.no_conform, aseg_save_path = args.output_aseg_path, conformed_input_save_path = args.output_T1_path, step_size = 32, device=device)
    del aseg_model

    if args.output_hsf_path is None:
        print('dont run hsf')
    else:
        hsf_model = MedSegMamba(img_dim = 96,
            patch_dim = 8,
            img_ch = 1,
            output_ch = 25,
            channel_sizes = [1,32,64,128,256,1024], 
            mamba_d_state = 64, # for vss block
            num_layers = 9, #6 or 9
            vss_version = 'v5', # None for vanilla
            mlp_dropout_rate = 0, #0.1
            attn_dropout_rate = 0, #0.1
            drop_path_rate=0, #0
            ssm_expansion_factor=1,
            id = False, #TABSurfer default
            preact = False, #TABSurfer default
            maxpool = True, #TABSurfer default
            upsample = True, #TABSurfer default
            full_final_block=True,
            scan_type='scan'
            )
        hsf_model.load_state_dict(torch.load(args.hsf_model_path, map_location=torch.device('cpu')))
        hsf_model.to(device)
        print('run hsf', args.hsf_model_path)
        hsf = run_hsf(hsf_model, aseg.get_fdata(), input_img, no_conform = args.no_conform, orig_shape = shape, orig_affine = affine, orig_header = header, hsf_save_path = args.output_hsf_path, device=device)
        del hsf_model

    print('done')
    #...
        