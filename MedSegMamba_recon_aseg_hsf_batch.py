#32577
#nohup /home/raphael/anaconda3/envs/SAIL/bin/python /media/sail/HDD18T/BME_Grad_Project/Aaron_DiffSurfer/medsegmamba_hsf/MedSegMamba_recon_HSF_batch.py  > /media/sail/HDD18T/BME_Grad_Project/Aaron_DiffSurfer/medsegmamba_hsf/run_on_Step2_3DDeepC_T1_MNI152affine_iso1mm.9.25.24.out &

#4288
#nohup /home/raphael/anaconda3/envs/SAIL/bin/python /media/sail/HDD18T/BME_Grad_Project/Aaron_DiffSurfer/medsegmamba_hsf/MedSegMamba_recon_HSF_batch.py  > /media/sail/HDD18T/BME_Grad_Project/Aaron_DiffSurfer/medsegmamba_hsf/run_on_Step4_3DDeepC_T1_MNI152affine_iso1mm_WM_Lin_Normalized.10.7.24.out &

import subprocess
import os
import pathlib
import argparse
import pandas as pd
from tqdm import tqdm
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default = 1000, type=int)
parser.add_argument('--input_dir', default='', type=str) 

# t1 and aseg paths are optional
parser.add_argument('--output_t1_dir', default=None, type=str) 
parser.add_argument('--output_aseg_dir', default=None, type=str)
parser.add_argument('--output_hsf_dir', default=None, type=str) 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser.add_argument('--no_conform', default=True, type=str2bool)

parser.add_argument('--aseg_model_path', default='', type=str)
parser.add_argument('--hsf_model_path', default='', type=str)

parser.add_argument('--python_env_path', default=None, type=str) 

parser.add_argument('--gpu_id', default=0, type=int) 
parser.add_argument('--aseg_step_size', default=32, type=int)

args = parser.parse_args()

if __name__ == '__main__':
    print(os.getpid())
    seed = args.seed

    input_dir = args.input_dir
    output_hsf_dir = args.output_hsf_dir

    output_T1_dir = args.output_t1_dir
    output_aseg_dir = args.output_aseg_dir

    aseg_model_path = args.aseg_model_path
    hsf_model_path = args.hsf_model_path

    gpu_id = args.gpu_id
    aseg_step_size = args.aseg_step_size
    no_conform = args.no_conform

    for file in tqdm(os.listdir(input_dir)):
        if file.endswith('.nii.gz') or file.endswith('.mgz'):
            #print(files)
            input_path = os.path.join(input_dir, file)
            file_name = file.split('.')[0]

            print('********************************************', flush = True)
            print('********************************************', flush = True)
            print(file_name, flush = True)
            print(input_path, flush = True)
            print(datetime.datetime.now(), flush=True)
            print('********************************************', flush = True)
            print('********************************************', flush = True)
            
            #continue
            new_args = ["--input_path", input_path, 
                    '--aseg_model_path', aseg_model_path,
                    '--hsf_model_path', hsf_model_path,
                    '--gpu_id', str(gpu_id),
                    '--aseg_step_size', str(aseg_step_size),
                    '--no_conform', str(no_conform),]
            
            if output_T1_dir is not None:
                if os.path.exists(output_T1_dir):
                    output_T1_path = os.path.join(output_T1_dir, f'T1.{file_name}.nii.gz')
                    new_args = new_args+["--output_T1_path", output_T1_path]
                else:
                    print('invalid output t1 folder:', output_T1_dir, flush=True)

            if output_aseg_dir is not None:    
                if os.path.exists(output_aseg_dir):
                    output_aseg_path = os.path.join(output_aseg_dir, f'aseg.{file_name}.nii.gz')
                    new_args = new_args+['--output_aseg_path', output_aseg_path]
                else:
                    print('invalid output aseg folder:', output_aseg_dir, flush=True)
            
            if output_hsf_dir is not None:    
                if os.path.exists(output_hsf_dir):
                    output_hsf_path = os.path.join(output_hsf_dir, f'hsf.{file_name}.nii.gz')
                    new_args = new_args+['--output_aseg_path', output_hsf_path]
                else:
                    print('invalid output hsf folder:', output_hsf_dir, flush=True)


            # Run the sub_script.py with arguments using subprocess
            if args.python_env_path is None:
                subprocess.run(["python", os.path.join(os.path.dirname(os.path.abspath(__file__)),"MedSegMamba_recon_HSF.py")] + new_args)
            else:
                subprocess.run([args.python_env_path, os.path.join(os.path.dirname(os.path.abspath(__file__)),"MedSegMamba_recon_HSF.py")] + new_args)

    print('done', flush = True)
