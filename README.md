# GPUMD-apptainer
A simple build of GPUMD with Apptainer to allow reliable, portable and reproducible use across systems and projects


### How to run container


docker run -it --privileged --runtime=nvidia  --gpus all -v $(pwd):/work --mount type=bind,src=/var/log/,dst=/var/log/
--mount type=bind,src=/etc/nvidia-container-runtime/,dst=/etc/nvidia-container-runtime/ --mount
type=bind,src=/usr/bin/nvidia-container-cli,dst=/usr/bin/nvidia-container-cli --mount
type=bind,src=/usr/bin/../lib/x86_64-linux-gnu/libnvidia-container.so.1,dst=/usr/bin/../lib/x86_64-linux-gnu/libnvidia-container.so.1
--mount
type=bind,src=/usr/lib/x86_64-linux-gnu/libnvidia-container-go.so.1,dst=/usr/lib/x86_64-linux-gnu/libnvidia-container-go.so.1
apptainer:1.4.4.deb13slim apptainer run --nvccli --bind /var/log/ --bind /etc/nvidia-container-runtime/ --bind /var/log/
--bind /usr/bin/ /work/gpumd.sif