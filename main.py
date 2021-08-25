import os
import logging
import sys
import framework 

logger = logging.getLogger(__name__)

'''
main.py train --name local --config example/configuration.yaml --tdata example/mock-image-label-pairs.npz --host local

python3.8 main.py generate --name peter --N 10 --host local

python3.8 main.py utility  --name man --host local --config example/configuration.yaml --patients example/patients/ --tdata seiton/man/synthetic/synthetic.npz

-------------------------------------
python3.8 main.py distance --man --host local --ref local/mock-image-label-pairs.npz

python3.8 main.py report --fullname

'''

if __name__ == '__main__':

    run_path_model = framework.RunPathBuilder(os.path.dirname(os.path.abspath(__file__))).build()
    
    [ logger.info(msg) for msg in run_path_model.validate_paths()]

    executor = framework.ModelExecutorBuilder(run_path_model).build()

    try:
        executor.run()
        logger.info('Au Revoir!')
        sys.exit(0)

    except Exception as e:
        print(e)
        logger.error(e,exc_info=1)
        logger.info(e,exc_info=1)
        sys.exit(-1)
    
