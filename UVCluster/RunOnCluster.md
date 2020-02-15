

This project have supported by the Lascilab Laboratory, and the next code are for run the prediction on a cluster with HTCondor Environment.

This project use a grass-py3-ml1 docker image, you can used downloading the image follow the next link: https://hub.docker.com/repository/docker/jsvinasco/grass-py3-ml1 

and run the process with the next command



sudo docker run -it --rm -v $(pwd):/grassdb/ jsvinasco/grass-py3-ml1 grass -text Tesis_Guajira_IA/PERMANENT  --exec bash prediccion_XGB.sh
