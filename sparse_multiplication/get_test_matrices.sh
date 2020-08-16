#!/usr/bin/env bash

mkdir -p data/

cd data/
wget ftp://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/bcspwr/bcspwr01.mtx.gz
gunzip bcspwr01.mtx.gz
wget ftp://math.nist.gov/pub/MatrixMarket2/SPARSKIT/drivcav/e05r0000.mtx.gz
gunzip e05r0000.mtx.gz
