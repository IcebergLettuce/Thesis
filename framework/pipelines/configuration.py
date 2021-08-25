import sys
import os
import argparse
import glob
import numpy as np
import logging

class RunPathModel():
    def __init__(self):

        self.host = None
        self.strategy = ''
        self.tdata_file = None
        
        self.patients_dir = None
        self.name = None
        self.config_file = None

        self.root_dir = None
        self.home_dir = None
        self.seiton_dir = None
        self.run_dir = None
        self.tensorboard_dir = None
        self.utility_dir = None
        self.utility_uiid_dir = None

        self.ref_data_file = None
        self.synthetic_dir = None
        self.log = None

        # model options
        self.weights = None 
        self.N = None

        # UID
        self.uiid = str(np.random.randint(0,1000,1)[0])
    
    def validate_paths(self):
        props = self.__dict__
        messages = []
        for k,v in props.items():
            if v == None:
                continue
            split = k.split('_')
            if len(split)>1:
                sp = split[1]
                
                if((sp == 'dir' )|(sp == 'file')):
                    try:
                        exists = os.path.exists(v)
                    except:
                        exists = False
                    if not exists:
                        messages.append(f'Missing files or directory structure: {k}:{v}')
                    else:
                        messages.append(f'Found file or directory: {k}:{v}')
        return messages

class RunPathBuilder():

    def __init__(self,root):

        if len(sys.argv[1:2]) == 0 :
            print('Provide valid command [train, generate, utility, report, distance]')
            sys.exit(1)

        self.path_model = RunPathModel()
        self.path_model.root_dir = root

        parser = argparse.ArgumentParser(description='Application shell that can do bla bla.')
        subparser = parser.add_subparsers(title='subcommands',help='additional help' , description='help')

        train_parser = subparser.add_parser('train')
        utility_parser = subparser.add_parser('utility')
        generate_parser = subparser.add_parser('generate')
        distance_parser = subparser.add_parser('distance')
        report_parser = subparser.add_parser('report')

        train_parser.add_argument('--name',help='provide name for run',required=True)
        train_parser.add_argument('--config',help='path to config file',required=True)
        train_parser.add_argument('--tdata',help='path to train data')
        train_parser.add_argument('--weights',help='path to weights to continue training',required=False)
        train_parser.add_argument('--host',help='host',required=True)
        train_parser.set_defaults(func = self._train)

        utility_parser.add_argument('--name',help='provide name for run',required=True)
        utility_parser.add_argument('--config',help='path to config file',required=True)
        utility_parser.add_argument('--tdata',help='path to train data',required=False)
        utility_parser.add_argument('--patients', help='path to patients data folder')
        utility_parser.add_argument('--host',help='host',required=True)
        utility_parser.set_defaults(func = self._utility)

        generate_parser.add_argument('--name',help='provide name for run',required=True)
        generate_parser.add_argument('--N',help='provide N as number of images to generate',default=1)
        generate_parser.add_argument('--host',help='host',required=True)
        generate_parser.set_defaults(func = self._generate)

        report_parser.add_argument('--name',help='provide name for run',required=True)
        report_parser.add_argument('--host',help='host',required=True)

        report_parser.set_defaults(func = self._report)

        distance_parser.add_argument('--name',help='provide name for run',required=True)
        distance_parser.add_argument('--ref',help='reference data',required=False)
        distance_parser.add_argument('--host',help='host',required=True)

        distance_parser.set_defaults(func = self._distance)

        args = parser.parse_args()
        args.func(args)

    
    def _create_paths(self):
        if self.path_model.host == 'clinic':
            self.path_model.home_dir = os.path.join(os.getenv('HOME'),'work')
        elif self.path_model.host == 'local':
            self.path_model.home_dir = self.path_model.root_dir
        
        self.path_model.seiton_dir = os.path.join(self.path_model.home_dir,'seiton')
        self.path_model.tensorboard_dir = os.path.join(self.path_model.home_dir,'seiton','tensorboard',self.path_model.name)
        self.path_model.run_dir = os.path.join(self.path_model.home_dir,'seiton',self.path_model.name)
        self.path_model.log = os.path.join(self.path_model.run_dir,'log.log')
        self.path_model.synthetic_dir = os.path.join(self.path_model.run_dir,'synthetic')

        self.path_model.utility = os.path.join(self.path_model.run_dir,'utility')
        self.path_model.utility_dir = os.path.join(self.path_model.run_dir,'utility')
        self.path_model.utility_uiid_dir = os.path.join(self.path_model.run_dir,'utility',self.path_model.uiid)

        self.path_model.weights = os.path.join(self.path_model.run_dir,'weights')

        if not os.path.exists(self.path_model.seiton_dir):
            os.mkdir(self.path_model.seiton_dir)

        if not os.path.exists(self.path_model.run_dir):
            os.mkdir(self.path_model.run_dir)

        if not os.path.exists(self.path_model.synthetic_dir):
            os.mkdir(self.path_model.synthetic_dir)

        if not os.path.exists(self.path_model.utility_dir):
            os.mkdir(self.path_model.utility_dir)
    
    def _configure_logging(self):
        logging.basicConfig(handlers=[
            logging.FileHandler(self.path_model.log),
            logging.StreamHandler()]
            ,level=logging.INFO
            ,format='%(asctime)s %(message)s'
            ,datefmt='%m/%d/%Y %H:%M:%S'
            )        

    def _distance(self,args):
        args = args.__dict__
        self.path_model.strategy  = 'distance'
        self.path_model.name = args['name']
        self.path_model.host = args['host']
        self.path_model.ref_data_file = args['ref']
        self._create_paths()
        self._configure_logging()

    def _report(self,args):
        args = args.__dict__
        self.path_model.strategy  = 'report'
        self.path_model.name = args['name']
        self.path_model.host = args['host']
        self._create_paths()
        self._configure_logging()


    def _train(self,args):
        args = args.__dict__
        self.path_model.strategy  = 'train'
        self.path_model.name = args['name']
        self.path_model.host = args['host']

        self.path_model.tdata_file = args['tdata']
        self.path_model.config_file = args['config']
        self._create_paths()
        self._configure_logging()


    def _generate(self,args):
        args = args.__dict__
        self.path_model.strategy  = 'generate'
        self.path_model.name = args['name']
        self.path_model.host = args['host']
        self.path_model.N = np.max([int(args['N']),10])
        self._create_paths()
        self.path_model.weights = os.path.join(self.path_model.run_dir,'weights')
        self.path_model.config_file = os.path.join(self.path_model.run_dir,'configuration.yaml')
        self._configure_logging()

    
    def _utility(self,args):
        args = args.__dict__
        self.path_model.strategy  = 'utility'
        self.path_model.name = args['name']
        self.path_model.host = args['host']

        self.path_model.patients_dir = args['patients']
        self.path_model.config_file = args['config']
        self._create_paths()

        if args['tdata'] == None:
            self.path_model.tdata_file = os.path.join(self.path_model.synthetic_dir,'synthetic.npz')
        else:
            self.path_model.tdata_file = args['tdata']
            
        self.path_model.tensorboard_dir = self.path_model.tensorboard_dir + '-unet-'+ str(self.path_model.uiid)

        if not os.path.exists(self.path_model.utility_dir):
            os.mkdir(self.path_model.utility_dir)

        os.mkdir(self.path_model.utility_uiid_dir)
        self._configure_logging()


    def build(self):
        return self.path_model

