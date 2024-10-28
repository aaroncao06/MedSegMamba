This is the official github repository for: https://arxiv.org/abs/2409.08307

The MedSegMamba architecture code is in the models folder. MedSegMamba is the model described in the paper, where the VSS3D modules are only present in the bottleneck. MedSegMamba_v2 has not been formally tested and allows VSS3D modules to replace skip connections between the encoder and decoder.

MedSegMamba_recon_aseg_hsf is the inference script for running subcortical or hippocampal subfield segmentation on just one scan. If you don't pass an argument for output_T1_path or output_aseg_path, the conformed input or subcortical segmentation will not be saved, respectively. If you don't pass an argument for output_hsf_path, it will not run the hippocampal subfield segmentation. aseg_step_size determines how much each patch the model is run on is shifted from one to another, so increasing it trades faster inference for possibly worse performance (in the MedSegMamba paper, we used a step size of 16 but 32 should work just fine). If no_conform is set to True, the output segmentation will match the orientation and location of the input image. If no_conform is set to False, the output segmentation will be conformed to LIA orientation and the new image will have dimensions 256x256x256 (matches the orientation and location of the conformed input image).

MedSegMamba_recon_aseg_hsf_batch runs MedSegMamba_recon_aseg_hsf on each .mgz or .nii.gz file in a folder and saves the outputs to other folders.

conform.py and arg_types.py come from the FastSurfer repository: https://github.com/Deep-MI/FastSurfer.

Model weights are available to download here: https://drive.google.com/drive/folders/1LFHNZqWiZJIwuUXZZsOQkuUCrNpzHIta?usp=sharing. It was trained on 1 mm isotropic resolution T1w MRI images preprocessed with FreeSurfer skull-stripping and intensity normalization.