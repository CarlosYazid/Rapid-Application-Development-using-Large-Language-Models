# Use a base image with Python
FROM nvcr.io/nvidia/tritonserver:25.10-py3-sdk

# Install necessary packages to add Docker repository and software-properties
# RUN apt-get update && apt-get install -y \
#     apt-utils ca-certificates curl gnupg lsb-release software-properties-common

# # Add Docker’s official GPG key
# RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# RUN install -m 0755 -d /etc/apt/keyrings
# RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
# RUN chmod a+r /etc/apt/keyrings/docker.asc

# # Set up the stable repository
# RUN echo \
#   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
#   $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# RUN apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin 

# Set working directory
WORKDIR /dli

RUN apt-get update && apt-get install -y apt-utils ffmpeg
RUN apt-get update && apt-get install -y build-essential bash curl unzip wget git
RUN apt-get update && apt-get install -y python3.12 python3-pip 
# RUN apt-get update && apt-get install -y nvidia-container-toolkit nvidia-cuda-toolkit

#Run pip dependencies
COPY composer/requirements.txt .
# RUN pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install -r requirements.txt

########################################################

# RUN pip3 uninstall -y pynvml && pip install --upgrade nvidia-ml-py
# RUN pip3 uninstall -y transformer-engine apex
# RUN pip3 uninstall -y flash-attn && pip install flash-attn
# RUN pip3 install opencv-fixer==0.2.5 && python -c "from opencv_fixer import AutoFix; AutoFix()"

RUN pip3 install --upgrade \
    ipywidgets==8.1.8 jupyterlab==4.6.4 jupyterlab-widgets==3.0.16 \
    jupyter-resource-usage==1.3.0 jupyter-archive==3.5.1 \
    jupytext==1.19.5 jupyterlab-nvdashboard==0.15.0

RUN pip3 install --upgrade \
    jupyterlab-language-pack-zh-CN \
    jupyterlab-language-pack-zh-TW

# RUN pip3 install jupyterlab-nvdashboard

COPY composer/jupyter/ /root/.jupyter
COPY composer/jupyter/bashrc /root/.bashrc
COPY composer/runtime/ /opt/course-runtime/
ENV PYTHONPATH=/opt/course-runtime:/dli/assessment/notebook_helpers
ENV SHELL /bin/bash

# COPY notebooks
WORKDIR /dli/task

# Expose port 8888 for JupyterLab
EXPOSE 8888 
EXPOSE 9012

# Start JupyterLab when the container runs
ADD composer/entrypoint.sh /usr/local/bin

ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]

# Keep the exact build recipe available in the learner workspace.
COPY composer/Dockerfile /opt/course-runtime/Dockerfile
