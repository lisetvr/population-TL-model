#============================================================
#
#  Data loader for volumes
#  If you use this motion model, please our work: 
#  "Predictive online 3D target tracking with population-based 
#  generative networks for image-guided radiotherapy" 
#  Accepted at IPCAI 2021
#
#  github id: lisetvr
#  MedICAL Lab
#============================================================

from torch.utils.data.dataset import Dataset
import os
import nibabel as nib
import torch
import torch.nn.functional as F
import numpy as np


class NAVIGATOR_4D_Dataset_img_seq(Dataset):

    def __init__(self, root_dir, nb_inputs=3, nb_pred=1, sequence_list=(), valid=False, test=False):

        self.root_dir = root_dir
        self.nb_inputs = nb_inputs
        self.nb_pred = nb_pred
        if len(sequence_list) != 0:
            sequences_dir = sequence_list
        else:
            sequences_dir = os.listdir(self.root_dir)
        self.files_input = list()
        self.files_output = list()
        self.ref_vol_files = {}
        self.exhale_as_reference = True
        self.test = test
        self.valid = valid
        self.ref_id = {'case01': 9, 'case02': 6, 'case03': 7, 'case04': 2, 'case05': 1, 'case06': 9, 'case07': 9,
                       'case08': 8, 'case09': 1, 'case10': 0, 'case11': 3, 'case12': 4, 'case13': 3, 'case14': 2,
                       'case15': 3, 'case16': 1, 'case17': 2, 'case18': 3, 'case19': 5, 'case20': 1, 'case21': 2,
                       'case22': 1, 'case23': 6, 'case24': 5, 'case25': 9}

        if self.test:
            temp_navs = 2
        elif self.valid:
            temp_navs = 10
        else:
            temp_navs = 20

        for sequence in sequences_dir:
            if sequence == ".directory":
                continue
            data_dir = os.path.join(self.root_dir, sequence)

            for z in range(temp_navs):
                for t in range(31 - (self.nb_inputs + self.nb_pred)):
                    img = list()
                    label = list()
                    for i in range(self.nb_inputs):
                        img.append(data_dir + '/t_' + str((31 * z + t) + i) + '.nii.gz')
                    for p in range(1, nb_pred + 1):
                        label.append(data_dir + '/t_' + str((31 * z + t) + (self.nb_inputs - 1) + p) + '.nii.gz')
                    self.files_input.append(img)
                    self.files_output.append(label)

            volume_files = os.listdir(data_dir)
            volume_files.sort(key=lambda x: int(x[2:-7]))
            ref_phase = self.ref_id[sequence]

            ref_vol_file = [file for file in volume_files if file.endswith("t_" + str(ref_phase) + ".nii.gz")][0]

            if not self.test: # When testing we want to make sure that the model does not move the reference volume
                self.files_input = [[ele for ele in sub if ele != ref_phase] for sub in self.files_input]
                indices = []
                for (ind, value) in enumerate(self.files_input):
                    if len(value) != self.nb_inputs:
                        indices.append(ind)
                self.files_input = [i for i in self.files_input if len(i) == self.nb_inputs]
                self.files_output = [j for (i, j) in enumerate(self.files_output) if i not in indices]

            self.ref_vol_files[sequence] = ("{}/{}".format(data_dir, ref_vol_file))

    def __len__(self):
        return len(self.files_input)

    def __getitem__(self, idx):
        # Load input volumes
        input_volume_list = list()
        for vol_file in self.files_input[idx]:
            input_volume = nib.load(vol_file).get_fdata()
            input_volume = (torch.from_numpy(input_volume)).float().unsqueeze(0).unsqueeze(0)
            input_volume = F.interpolate(input_volume, scale_factor=[1, 0.5, 0.5], mode='trilinear').squeeze()
            input_volume = (input_volume - torch.mean(input_volume)) / torch.std(input_volume)
            input_volume_list.append(input_volume)

        # Load output volume
        for vol_file in self.files_output[idx]:
            output_volume = nib.load(vol_file).get_fdata()
            output_volume = (torch.from_numpy(output_volume)).float().unsqueeze(0).unsqueeze(0)
            output_volume = F.interpolate(output_volume, scale_factor=[1, 0.5, 0.5], mode='trilinear').squeeze()
            output_volume = (output_volume - torch.mean(output_volume)) / torch.std(output_volume)

        # Load reference volume
        ref_vol_file = self.ref_vol_files[self.files_input[idx][0].split('/')[-2]]
        ref_volume = nib.load(ref_vol_file).get_fdata()
        ref_volume = (torch.from_numpy(ref_volume)).float().unsqueeze(0).unsqueeze(0)
        ref_volume = F.interpolate(ref_volume, scale_factor=[1, 0.5, 0.5], mode='trilinear').squeeze()
        ref_volume = (ref_volume - torch.mean(ref_volume)) / torch.std(ref_volume)

        return ref_volume, input_volume_list, output_volume, self.files_output[idx]
