#!/bin/bash
set -e
mkdir -p data
curl -sL -o data/nslkdd_train.csv "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
curl -sL -o data/nslkdd_test.csv "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
curl -sL -o repo_unsw.zip "https://codeload.github.com/abhinav-bhardwaj/IoT-Network-Intrusion-Detection-System-UNSW-NB15/zip/refs/heads/master"
unzip -p repo_unsw.zip "IoT-Network-Intrusion-Detection-System-UNSW-NB15-master/datasets/UNSW_NB15.csv" > data/unsw.csv
rm -f repo_unsw.zip
curl -sL -o data/edge_iiotset.csv "https://raw.githubusercontent.com/Rayan-Ali1083/Layer7Defend/main/ML-EdgeIIoT-dataset.csv"
echo "Done. To add TON_IoT, download train_test_network.csv from https://research.unsw.edu.au/projects/toniot-datasets and place it at data/ton_iot.csv"
