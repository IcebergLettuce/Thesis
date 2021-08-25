# Content

This repository contains the source code and the end-to-end pipeline of the Thesis: "xxx.". The system was developed to run on the High-Performance Cluster of the Berlin Institute of Health to automate experiments. This system allowed to run many experiments with as little manual overhead as possible.

The original data used in the study is a property of Charite and, unfortunately, publicly not available. However, we made it possible to experience the end-to-end pipeline locally within a Docker container. The data we used and offer is mock data but still allows you to run the entire pipeline.

We emphasize running the Docker example on a local machine and experience the framework and how it works. We hope that this - still - sketchy implementation motivates people to automate their workflows similarly.

**Improvements and Directions:**\
We suggest splitting the framework into multiple stand-alone applications (microservices) and maintain them in their repositories. They would be served as APIs to the user who develops their model against these APIs. Additionally, we recommend using a proper relational database instead of the filesystem.


**This repository contains the following:**

- Do-It-Yourself: docker image and explanation for running the end-to-end pipeline by yourself without having access to the HPC and without the struggle of installing dependencies.
- Source code of the thesis "xyz".
- Source code of the used computational framework.

# Running the end-to-end pipeline

One can run the full end-to-end pipeline locally and get an impression of how the framework works. Just follow this simple guide:

**Requirements:**
- [Docker](https://www.docker.com/)

**How To:**
1. Navigate to your home directory ``cd ~ ``
2. Create a folder with the name example ``mkdir example ``
3. Pull the docker image with ``docker pull hirzeman/seiton:example``
4. Run the container with ``docker run --mount type=bind,source=$HOME/example,target=/app/seiton -it hirzeman/seiton:example /bin/bash``

**What will happen?**

This docker container contains this repository and some sample data from the folder example. One can simulate a big run locally and see how the pipeline would behave on the large cluster. The pipleine trains a very very small gan with image-label pairs, generates synthetic image-label pairs, calculates the utility by training the U-Net, evaluates the U-Net on a patient, calculates the distances between synthetic and real data, and finally genereates a computing report\
\
By opening the folder ``example`` one can see the results of the computing pipeline. There should be a folder named reports containing the computing report of your run. You can ope this report with any pdf-viewer. Also interesting is the log.log file which keeps detailed logging info of your experiment.\
\
**Fun stuff**

You can also overwrite the used run configuration with a local one and modify the training and parameters of the pipeline. Just copy-paste the ``configuration.yaml`` of this repository into your local ``example`` folder, and modify some numbers.\
Use the following command to modify it within the container:
```
docker run --mount type=bind,source=$HOME/example,target=/app/seiton --mount type=bind,source=$HOME/example/configuration.yaml,target=/app/example/configuration.yaml -it hirzeman/seiton:latest /bin/bash
```

# Source code framework
Using the framework's source code is quite tricky, as the framework was developed against the HPC BIH infrastructure. There is still much refactoring required to make the approach more general-purpose and eligible for different use cases. Overall, the idea was to pipeline different steps of computations in an easy and automated way and to be able to create computation summaries in the form of reports. As one can see the in the `pipeline.sh` file the idea is to split everything up into smaller, easier to control parts:

> python main.py train --name myrun --config example/configuration.yaml --tdata example/mock-image-label-pairs.npz --host local \
  python main.py generate --name myrun -N 100 --host local \
. \
. \
python main.py report --name myrun --host local

\
The results are then synchronized with the generated filesystem. In a more elaborated approach we would strongly suggest to build the framework against APIs and Databases.

# Source code components
The used components like neural networks, GANs, metrics etc. can be found in the sub-folder framework/components. Generally, it should be possible to copy-paste the pieces and re-use them in own work. 
