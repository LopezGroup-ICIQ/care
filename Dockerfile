# Use an official Python 3.11 base image
FROM python:3.11-slim

# Set environment variables
ENV CONDA_ENV=care_env
ENV LANG C.UTF-8

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init bash

# Set PATH for conda
ENV PATH /opt/conda/bin:$PATH

# Copy all files (except the specified in .dockerignore) from the repo to the /care directory in the container
COPY . /care

# Set working directory
WORKDIR /care

# Create and activate conda environment
RUN conda create -n $CONDA_ENV python=3.11 -y && \
    echo "conda activate $CONDA_ENV" >> ~/.bashrc

# Install Python dependencies
RUN /opt/conda/bin/conda run -n $CONDA_ENV python3 -m pip install .

# Install pytorch and pytorch_geometric dependencies
RUN /opt/conda/bin/conda install pytorch cpuonly pytorch-scatter pytorch-sparse pyg -c pytorch -c pyg -y

# Install Julia (optional) with non-interactive install
RUN curl -fsSL https://install.julialang.org | sh -s -- -y && \
    /opt/conda/bin/conda run -n $CONDA_ENV python3 -m pip install juliacall && \
    . /root/.bashrc && \
    . /root/.profile && \
    julia -e 'import Pkg; Pkg.add("DifferentialEquations"); Pkg.add("DiffEqGPU"); Pkg.add("CUDA");'

# Install fairchem and mace-torch (optional)
RUN /opt/conda/bin/conda run -n $CONDA_ENV python3 -m pip install fairchem-core==1.1.0 && \
    /opt/conda/bin/conda run -n $CONDA_ENV python3 -m pip install mace-torch torch-dftd

# Add binaries from the project into PATH
ENV PATH $PATH:/opt/conda/envs/care_env/bin

# Copy logo into required place as required by hard-coded relative path
RUN cp /care/src/care/logo.txt /opt/conda/envs/$CONDA_ENV/lib/python3.11/site-packages/care/

# Create folder for data volume mount point
RUN mkdir -p /data

# Clean build artifacts from image
RUN rm -rf /care

# Set the entrypoint to run the gen_crn_blueprint command
ENTRYPOINT ["gen_crn_blueprint", "-cs", "CCO", "C(CO)O", "-o", "/data/output_name"]

# Default command (you can override this when running the container)
CMD []