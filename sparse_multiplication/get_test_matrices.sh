#!/usr/bin/env bash

mkdir -p data/

cd data/
wget ftp://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/counterx/jgl009.mtx.gz
gunzip jgl009.mtx.gz
wget ftp://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/counterx/rgg010.mtx.gz
gunzip rgg010.mtx.gz
