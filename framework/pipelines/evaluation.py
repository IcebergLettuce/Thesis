import os
import glob
import nibabel as nib
import numpy as np
import time
import json 
import re

import logging

logger = logging.getLogger(__name__)

class Result:
    def __init__(self):
        self.truth = None
        self.pred = None
        self.xmlpath = None

class Results:
    def __init__(self):
        self.epoch = None
        self.train = [] # List of Results
        self.test = [] # List of Results
        self.val = [] # List of Results

# create a validation metrics file withion the utitliy/no folder
# executed only once at the end
class Evaluation:
    def __init__(self, smodel):
        self.udict = smodel.run_path_model.utility_uiid_dir#os.path.join(smodel.run_path_model.utility_dir,'561')
        self.exec_path = os.path.join(smodel.run_path_model.root_dir,'EvaluateSegmentation')#'/data/users/kossent/work/EvaluateSegmentation'
        self.patients_dir = smodel.run_path_model.patients_dir
        logger.info('Initialised and registered Segmentation Evaluation')

    def process(self):
        
        commands = self._generate_commands(self._collect_data())
        for command in commands:
            logger.info(f'Running: {command}')
            os.system(command)

    def _generate_commands(self,files):
        
        commands = []
        threshold = 0.5
        measures = 'DICE,HDRFDST@0.95@'
        for file in files:
            logger.info(f'Creating command for epoch: {file.epoch}')
            for tr in file.train:
                logger.info(f'Creating command for train files.')
                commands.append(f'{self.exec_path} {tr.truth} {tr.pred} -use {measures} -xml {tr.xmlpath} -thd {threshold} -unit millimeter')

            for ts in file.test:
                logger.info(f'Creating command for test files.')
                commands.append(f'{self.exec_path} {ts.truth} {ts.pred} -use {measures} -xml {ts.xmlpath} -thd {threshold} -unit millimeter')

            for val in file.val:
                logger.info(f'Creating command for val files.')
                commands.append(f'{self.exec_path} {val.truth} {val.pred} -use {measures} -xml {val.xmlpath} -thd {threshold} -unit millimeter')
        
        return commands


    def _collect_data(self):
        
        epochs = os.listdir(self.udict)
        ft = re.compile(r'nii')
        results_collection = []

        for epoch in epochs:
            
            pp = os.path.join(self.udict,epoch)
            if not os.path.basename(pp).isdigit():
                continue

            results = Results()
            results.epoch = os.path.basename(pp)

            if os.path.exists(os.path.join(pp,'train')):
                train_files = os.listdir(os.path.join(pp,'train')) 
                train_files = [ s for s in train_files if ft.search(s) ]
                
                train_target = os.path.join(pp,'train','results')
                os.makedirs(train_target,exist_ok= True)
                for train_f in train_files:
                    r = Result()
                    r.pred = os.path.join(os.path.join(pp,'train'),train_f)
                    r.truth = self._find_goldstandard(train_f)
                    r.xmlpath = os.path.join(train_target,os.path.basename(train_f)+'_.xml')
                    results.train.append(r)

            if os.path.exists(os.path.join(pp,'val')):
                val_files= os.listdir(os.path.join(pp,'val'))
                val_files = [ s for s in val_files if ft.search(s) ]

                val_target = os.path.join(pp,'val','results')
                os.makedirs(val_target,exist_ok= True)
                for val_f in val_files:
                    r = Result()
                    r.pred = os.path.join(os.path.join(pp,'val'),val_f)
                    r.truth = self._find_goldstandard(val_f)
                    r.xmlpath = os.path.join(val_target,os.path.basename(val_f)+'_.xml')
                    results.train.append(r)

            if os.path.exists(os.path.join(pp,'test')):
                test_files = os.listdir(os.path.join(pp,'test')) 
                test_files = [ s for s in test_files if ft.search(s) ]

                test_target = os.path.join(pp,'test','results')
                os.makedirs(test_target,exist_ok= True)
                for test_f in test_files:
                    r = Result()
                    r.pred = os.path.join(os.path.join(pp,'test'),test_f)
                    r.truth = self._find_goldstandard(test_f)
                    r.xmlpath = os.path.join(test_target,os.path.basename(test_f)+'_.xml')
                    results.train.append(r)
            
            results_collection.append(results)
        return results_collection
        
    def _find_goldstandard(self, name):
        logger.info('Searching for gold-standard scan of patient.')
        fname = os.path.basename(name)
        sfor = fname.split('_')[0]
        a = glob.glob(os.path.join(self.patients_dir,'val',sfor+'_label*'))
        b = glob.glob(os.path.join(self.patients_dir,'train',sfor+'_label*'))
        c = glob.glob(os.path.join(self.patients_dir,'test',sfor+'_label*'))
        d = a+b+c
        return d[0]

class PredictionGenerator:
    def __init__(self, smodel,dataset):
        self.dataset = dataset
        self.smodel = smodel
        self.mean = smodel.train_data_mean
        self.std = smodel.train_data_std
        self.fpath = os.path.join(smodel.run_path_model.patients_dir,dataset)
        self.img_files = glob.glob(os.path.join(self.fpath,'*img.nii*'))
        self.mask_files = glob.glob(os.path.join(self.fpath,'*mask.nii*'))
        
        if(len(self.img_files)!=len(self.mask_files)):
            raise Exception('Number of image and mask files does not match!')
        logger.info('Initialised and registered PredictionGenerator')


    def process(self,epoch):
        logger.info('Start generating predictions')
        patch_size = 96
        mean = self.mean
        sd = self.std
        
        directory = os.path.join(self.smodel.run_path_model.utility_uiid_dir,str(epoch),self.dataset)
        os.makedirs(directory)
        for image in self.img_files:
            savefile = directory + '/' + image.split('/')[-1].split('_')[0]+'_label.nii'
            starttime_total = time.time()

            img_mat = nib.load(image).get_fdata()
            mask_mat = nib.load(self._find_mask(image)).get_fdata()

            logger.info('Processing image: ' + str(image))
            logger.info('Processing mask: ' + str(self._find_mask(image)))
            # -----------------------------------------------------------
            # PREDICTION
            # -----------------------------------------------------------
            # the segmentation is going to be saved in this probability matrix
            prob_mat = np.zeros(img_mat.shape, dtype=np.float32)
            x_dim, y_dim, z_dim = prob_mat.shape

            # get the x, y and z coordinates where there is brain
            x, y, z = np.where(mask_mat)

            # get the z slices with brain
            z_slices = np.unique(z)

            # start cutting out and predicting the patches
            starttime_total = time.time()
            # proceed slice by slice
            for i in z_slices:
                slice_vox_inds = np.where(z == i)
                # find all x and y coordinates with brain in given slice
                x_in_slice = x[slice_vox_inds]

                y_in_slice = y[slice_vox_inds]

                # find min and max x and y coordinates
                slice_x_min = min(x_in_slice)
                slice_x_max = max(x_in_slice)

                slice_y_min = min(y_in_slice)
                slice_y_max = max(y_in_slice)
              
                if(np.abs(slice_x_max-slice_x_min))<96:
                    continue
                
                if(np.abs(slice_y_max-slice_y_min))<96:
                    continue

              

                # calculate number of predicted patches in x and y direction
                # in given slice
                num_of_x_patches = np.int(np.ceil((slice_x_max - slice_x_min)
                                                  / patch_size))
                num_of_y_patches = np.int(np.ceil((slice_y_max - slice_y_min)
                                                  / patch_size))

                # predict patch by patch in given slice
                for j in range(num_of_x_patches):
                    for k in range(num_of_y_patches):
                        # find the starting and ending x and y coordinates of
                        # given patch
                        patch_start_x = slice_x_min + patch_size * j
                        patch_end_x = slice_x_min + patch_size * (j + 1)
                        patch_start_y = slice_y_min + patch_size * k
                        patch_end_y = slice_y_min + patch_size * (k + 1)
                        # if the dimensions of the probability matrix are
                        # exceeded shift back the last patch
                        if patch_end_x > x_dim:
                            patch_end_x = slice_x_max
                            patch_start_x = slice_x_max - patch_size
                        if patch_end_y > y_dim:
                            patch_end_y = slice_y_max
                            patch_start_y = slice_y_max - patch_size

                        # get the patch with the found coordinates from the
                        # image matrix

                        img_patch = img_mat[patch_start_x: patch_end_x,
                                    patch_start_y: patch_end_y, i]

                        # normalize the patch with mean and standard deviation
                        # calculated over training set
                        img_patch = img_patch.astype(np.float)
                        img_patch -= mean
                        img_patch /= sd

                        # predict the patch with the model and save to
                        # probability matrix

                        prob_mat[patch_start_x: patch_end_x,
                        patch_start_y: patch_end_y, i] = np.reshape(
                            self.smodel.model.predict(np.reshape(img_patch,
                                                     (1, patch_size,
                                                      patch_size, 1)),
                                          batch_size=1, verbose=0),
                            (patch_size, patch_size))
            
            new_nifti = nib.Nifti1Image(prob_mat, np.eye(4))  # create new nifti from matrix
            nib.save(new_nifti,savefile)
            logger.info('Writing to: ' + str(savefile))

            # how long does the prediction take for a patient
            duration_total = time.time() - starttime_total
            logger.info('Processing duration: ' + str(np.round(duration_total,3)))    

    def _find_mask(self,image_name):
        for mask in self.mask_files:
            if mask.split('_')[0] == image_name.split('_')[0]:
                return mask
        return None
            