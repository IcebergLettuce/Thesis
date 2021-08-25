import argparse
import yaml
import glob
import shutil
import numpy as np
import os
import ot
import copy
import shutil
from tensorflow.keras.callbacks import TerminateOnNaN,ReduceLROnPlateau, CSVLogger, TensorBoard, ModelCheckpoint
from sklearn.neighbors import NearestNeighbors
from .components.builder import ModelBuilder
from .pipelines.reporting import *
from .components.provider import SyntheticDataProvider,ImageLabelProvider
from .pipelines.extensions import ImageGeneratorCallback, TrainTestCallback, ValidationCallback,PyTorchImageGeneratorCallback, EvaluateSegmentationCallback
from .pipelines.evaluation import PredictionGenerator, Evaluation
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.mixture import BayesianGaussianMixture
import logging

logger = logging.getLogger(__name__)


class ExecutionModel:    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.validation_data = None
        self.model = None
        self.config = None
        self.run_path_model = None
        self.hp = None
        self.logger = None
        self.train_data_mean = None
        self.train_data_std = None

class ModelExecutorBuilder:

    def __init__(self, run_path_model):
        
        logger.info('Initialising model executor.')
        self.exec_model = ExecutionModel()
        self.exec_model.run_path_model = run_path_model

        if run_path_model.strategy == 'distance':
            logger.info('Initialising distance strategy.')
            self.strategy = DistanceStrategy(self.exec_model)
            return 

        if run_path_model.strategy == 'report':
            logger.info('Initialising report strategy.')
            self.strategy = ReportStrategy(self.exec_model)
            return

        with open(run_path_model.config_file) as file:
            self.exec_model.config = yaml.load(file)

        if run_path_model.strategy == 'train':
            logger.info('Initialising train strategy.')
            self.exec_model.hp = self.exec_model.config['hp']
            self.exec_model.model = ModelBuilder.build(self.exec_model.hp)

            if self.exec_model.hp['architecture'] == 'DPWGAN':
                logger.info('Initialising DPWGAN image label provider with a TorchDataset')
                self.exec_model.train_data = ImageLabelProvider(run_path_model.tdata_file).get('TorchDataset')

            elif self.exec_model.hp['architecture'] == 'MINIGAN':
                logger.info('Initialising MINIGAN image label provider with a TorchDataset')
                self.exec_model.train_data = ImageLabelProvider(run_path_model.tdata_file).get('TorchDataset')

            else:
                logger.info('Initialising OTHER image label provider with a numpy dataset')
                self.exec_model.train_data = ImageLabelProvider(run_path_model.tdata_file).get()
                self.exec_model.train_data = self.exec_model.train_data.batch(self.exec_model.hp['batch_size']).prefetch(self.exec_model.hp['batch_size']).shuffle(self.exec_model.train_data.cardinality(), reshuffle_each_iteration=True)
            
            self.strategy = TrainStrategy(self.exec_model)

            logger.info('Checking for weights to continue training.')
            if len(glob.glob(run_path_model.weights+'*')) > 0 :
                logger.info('Weighst were found.')
                self.exec_model.model.load_weights(run_path_model.weights)

                logger.info('Weights were laoded to the model.')
                shutil.rmtree(self.exec_model.run_path_model.tensorboard_dir)

            logger.info('Save configuration.yaml into the run folder')
            with open(os.path.join(run_path_model.run_dir,'configuration.yaml'), 'w') as file:
                yaml.dump(self.exec_model.config, file)

        if run_path_model.strategy == 'generate':
            logger.info('Initialising generate strategy.')
            self.exec_model.hp = self.exec_model.config['hp']
            self.exec_model.model = ModelBuilder.build(self.exec_model.hp)
            self.strategy = GenerateStrategy(self.exec_model)
        
        if run_path_model.strategy == 'utility':
            logger.info('Initialising utility strategy.')
            self.exec_model.hp = self.exec_model.config['hputility']
            self.exec_model.model = ModelBuilder.build(self.exec_model.hp)

            self.exec_model.train_data,self.exec_model.train_data_mean, self.exec_model.train_data_std = SyntheticDataProvider(run_path_model.tdata_file).get()
            self.strategy = UtilityStrategy(self.exec_model)

            logger.info('Save configuration.yaml into the utility run folder.')
            with open(os.path.join(run_path_model.utility_uiid_dir,'configuration.yaml'), 'w') as file:
                yaml.dump(self.exec_model.config, file)

    def build(self):
        logger.info('ModelExecutorBuilder was successfull.')
        return self.strategy

class BaseStrategy:
    def __init__(self, smodel):
        self.smodel = smodel
        self.config = smodel.config
        self.model = smodel.model

class TrainStrategy(BaseStrategy):
    def __init__(self,smodel):
        super().__init__(smodel)
        pass

    def run(self): 
        self.model.fit(self.smodel.train_data,
            self.smodel.config['model']['epochs'], 
            self.smodel.run_path_model.run_dir, 
            PyTorchImageGeneratorCallback())


class GenerateStrategy(BaseStrategy):
    def __init__(self,smodel):
        super().__init__(smodel)
        pass
    
    def run(self): 
        gan_img, gan_label = self.smodel.model.generate(self.smodel.run_path_model.run_dir, self.smodel.run_path_model.N)
        self._save_files(gan_img, gan_label)
        self._save_images(gan_img,gan_label)

    def _save_files(self, gan_img, gan_label):
        logger.info('Saving synthetic data.')
        np.savez_compressed(
            os.path.join(self.smodel.run_path_model.run_dir,'synthetic','synthetic.npz'),
            img=gan_img[:, :, :, np.newaxis],
            label=gan_label[:, :, :, np.newaxis],
        )

        logger.info('Saving a smaller subset of synthetic data.')
        np.savez_compressed(
            os.path.join(self.smodel.run_path_model.run_dir,'synthetic','synthetic-small.npz'),
            img=gan_img[:1000, :, :, np.newaxis],
            label=gan_label[:1000, :, :, np.newaxis],
        )

    def _save_images(self,gan_img, gan_label):
        
        fig, ax = plt.subplots(ncols=3, nrows=5, constrained_layout=True,figsize=(5, 10))
        fig.suptitle(f'Synthetic Images', fontsize=16)

        for nrow,col in enumerate(ax):
            for ncol, axc in enumerate(col):

                axc.set_xticklabels([])
                axc.set_yticklabels([])
                axc.axis('off')
                axc.set_aspect('equal')

                image = gan_img[nrow,:,:]
                label = gan_label[nrow,:,:]

                label[label>0.5] = 1.0
                label[label<= 0.5] = 0.0

                image =  (image - image.min()) / (image.max() - image.min())

                if ncol == 0:
                    axc.imshow(image, cmap='gray')
                
                if ncol == 1:
                    axc.imshow(label, cmap='gray')

                if ncol == 2:
                    axc.imshow(image, cmap='gray',interpolation='none')
                    masked = np.ma.masked_where(label < 0.5, label)
                    axc.imshow(masked, cmap='coolwarm_r', alpha=0.5,interpolation='none') # interpolation='none'

                if nrow == 0:
                    if ncol ==0:
                        axc.set_title("Image")
                    if ncol ==1:
                        axc.set_title("Label")
                    if ncol ==2:
                        axc.set_title("Combined")

        fig.savefig(self.smodel.run_path_model.synthetic_dir+'/synthetic.jpg', bbox_inches='tight',pad_inches = 0)
        plt.close()

class UtilityStrategy(BaseStrategy):
    def __init__(self,smodel):
        super().__init__(smodel)
        pass

    def run(self):
        callbacks = []
        
        callbacks.append(ValidationCallback(PredictionGenerator(self.smodel,'val')))
        callbacks.append(TrainTestCallback(PredictionGenerator(self.smodel,'test')))
        callbacks.append(TrainTestCallback(PredictionGenerator(self.smodel,'val')))
        callbacks.append(TrainTestCallback(PredictionGenerator(self.smodel,'train')))
        callbacks.append(EvaluateSegmentationCallback(Evaluation(self.smodel)))

        callbacks.append(CSVLogger(os.path.join(self.smodel.run_path_model.utility_uiid_dir,'training.log')))
        callbacks.append(TensorBoard(self.smodel.run_path_model.tensorboard_dir, histogram_freq = 0))

        self.smodel.model.fit(self.smodel.train_data, steps_per_epoch= 1,callbacks = callbacks, epochs = 3) 
        a = Evaluation(self.smodel)
        a.process()

class ReportStrategy(BaseStrategy):

    def __init__(self,smodel):
        super().__init__(smodel)
        logger.info('Creatig a report by collecting data from run filesystem.')

    def run(self):
        model = SummaryModel()
        model.name = self.smodel.run_path_model.name
        model.base_path = self.smodel.run_path_model.seiton_dir
        model.project_path = os.path.join(model.base_path,model.name)

        if not os.path.exists(model.project_path):
            logger.error('Project does not exist.')
            sys.exit(0)

        random = np.random.randint(0,1000)

        logger.info('Creatig temproy folders and files to cache data.')
        if not os.path.exists(f'sumtmp{random}'):
            os.mkdir(f'sumtmp{random}')
        
        if not os.path.exists(os.path.join(model.base_path,'reports')):
            os.mkdir(os.path.join(model.base_path,'reports'))
        
        model.tmp_path = f'sumtmp{random}'
        model.generate_path_structure()
        model.generate_html()
        shutil.rmtree(f'sumtmp{random}')

'''
DeprecationWarning
'''
class DistanceStrategy(BaseStrategy):

    def __init__(self,smodel):
        super().__init__(smodel)        

    def run(self):
        logger.info('Running the distance strategy to calculate the similarity between datases.')

        def load(path):
            data = np.load(path)
            img = data['img']
            lab = data['label']
            return (img,lab)
        

        r_img,r_label= load(self.smodel.run_path_model.ref_data_file)
        ref = np.concatenate((r_img,r_label),axis=3)
        #ref = ref[np.random.choice(range(0,ref.shape[0]),sample_size)]
        reff = copy.deepcopy(ref)
        ref_ = np.reshape(ref,(ref.shape[0],-1))
        

        sample_size = 100

        g_img,g_label=load(os.path.join(self.smodel.run_path_model.synthetic_dir,'synthetic.npz'))
        
        gen = np.concatenate((g_img,g_label),axis=3)
        gen = gen[np.random.choice(range(0,gen.shape[0]),sample_size)]
        genn = copy.deepcopy(gen)
        gen_ = np.reshape(gen,(gen.shape[0],-1))    

        path = os.path.join(self.smodel.run_path_model.synthetic_dir,'distances')
        os.makedirs(path,exist_ok = True)

        def get_sorted_index(d_mat):
            in_ = d_mat.shape[0]
            jn_ = d_mat.shape[1]
            idx = np.argsort(d_mat.flatten())
            i_ = idx // jn_
            j_ = idx - i_*jn_
            return np.concatenate((i_.reshape(-1,1),j_.reshape(-1,1)),axis=1)

        def normal(x,mu,sigma):
            return (1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (x - mu)**2 / (2 * sigma**2) ))

        def to_histograms(ref):
            result = []
            for img in ref:
                hist, edges = np.histogram(img.flatten(), bins=256, range=(0, 255))
                result.append(hist+1) # Laplace Correction
            return np.asarray(result)

        gen_histos = to_histograms(gen_) # List of gen histograms
        ref_histos = to_histograms(ref_) # List of histograms
        
        logger.info('Calculate historgrams.')

        
        distance_matrix = np.ones((gen_histos.shape[0],ref_histos.shape[0]))
        logger.info('Initialized Distance matrix')
        logger.info('Start calculating wasserstein distance.')
        for i,gen in enumerate(gen_histos):
            #print(f'Running {i} of {len(gen_histos)}')
            for j,ref in enumerate(ref_histos):
                n = 256
                M = ot.dist(gen.reshape((n, 1)), gen.reshape((n, 1)))
                M = M / M.max()
                lambd = 1e-3
                Gs, log = ot.emd(ref, gen, M, log = True)
                distance_matrix[i,j] = log['cost']

        logger.info('Distance matrix calculated.')
        dist = distance_matrix.flatten()

        x = np.linspace(np.min(dist),np.max(dist),1000)
        normals = []
        components = []

        logger.info('Plot distances as histogram.')
        fig, ax = plt.subplots(ncols=1,nrows=1)
        ax.hist(dist,1000, density = True)
        # for elem in normals:
        #     ax.plot(x,elem[0], label = f'mu= {np.round(elem[1],3)} std= {np.round(elem[2],3)} comp= {elem[3]}')

        ax.set_title('Distribution of KL divergences between images')
        ax.set_xlabel('EM Distances')
        ax.set_ylabel('Frequency')
        # plt.legend()
        plt.savefig(os.path.join(path,'hist.jpg'))
        plt.close()

        sorted_idx = get_sorted_index(distance_matrix)

        for i in range(10):
            fig, ax = plt.subplots(ncols=4, nrows=1, constrained_layout=True,figsize=(10, 3))
            fig.suptitle(f'closest nearest neighbors of dataset', fontsize=16)

            for nrow,axc in enumerate(ax):
                axc.set_xticklabels([])
                axc.set_yticklabels([])
                axc.axis('off')
                axc.set_aspect('equal')
                if nrow == 0:
                    axc.imshow(reff[sorted_idx[i][1],:,:,0],cmap='gray')
                    axc.set_title('Truth Image')
                if nrow == 1:
                    axc.imshow(genn[sorted_idx[i][0],:,:,0],cmap='gray')
                    axc.set_title('Synthetic Image')
                if nrow == 2:
                    axc.imshow(reff[sorted_idx[i][1],:,:,1],cmap='gray')
                    axc.set_title('Truth Label')
                if nrow == 3:
                    axc.imshow(genn[sorted_idx[i][0],:,:,1],cmap='gray')
                    axc.set_title('Synthetic Label')
        #    plt.savefig(f'{path}/closest{i}.png')
            plt.savefig(os.path.join(path,f'closest{i}.png'))
            plt.close()

        logger.info('Randomly select neigboring images to be added to the report.')
        for idx,comp in enumerate(components):
            search = True
            N = 3
            s_count = 0
            while search:
                s_i = np.random.choice(range(distance_matrix.shape[0]))
                s_j = np.random.choice(range(distance_matrix.shape[1]))

                pred = bgm.predict([[distance_matrix[s_i,s_j]]])[0]
                if pred == comp:

                    fig, ax = plt.subplots(ncols=4, nrows=1, constrained_layout=True,figsize=(10, 3))
                    fig.suptitle(f'Neigbours by component {comp}', fontsize=16)

                    for nrow,axc in enumerate(ax):
                        axc.set_xticklabels([])
                        axc.set_yticklabels([])
                        axc.axis('off')
                        axc.set_aspect('equal')
                        if nrow == 0:
                            axc.imshow(reff[s_j,:,:,0],cmap='gray') # Add index!!!!
                            axc.set_title('Truth Image')
                        if nrow == 1:
                            axc.imshow(genn[s_i,:,:,0],cmap='gray')
                            axc.set_title('Synthetic Image')
                        if nrow == 2:
                            axc.imshow(reff[s_j,:,:,1],cmap='gray')
                            axc.set_title('Truth Label')
                        if nrow == 3:
                            axc.imshow(genn[s_i,:,:,1],cmap='gray')
                            axc.set_title('Synthetic Label')
                #    plt.savefig(f'{path}/closest{i}.png')
                    plt.savefig(os.path.join(path,f'sample_{s_count}_by_comp_{comp}.png'))
                    plt.close()
                    s_count = s_count + 1
                
                if s_count == N:
                    search = False

        np.savez(os.path.join(path,'data.npz'),distances = distance_matrix, sorted_idx = sorted_idx)
        logger.info('Dumped distances as numpy array to filesystem.')
        logger.info('Distance stragey done.')


        