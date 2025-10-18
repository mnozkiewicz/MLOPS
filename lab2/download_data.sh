#!/bin/bash

prefix=https://opendata.rijdendetreinen.nl/public

wget $prefix/tariff-distances/tariff-distances-2022-01.csv -O data/inter-station-2022.csv
wget $prefix/stations/stations-2023-09-nl.csv -O data/stations-2023.csv

for year in {2011..2023}; 
do 
    wget $prefix/disruptions/disruptions-$year.csv -O data/disruptions-$year.csv; 
done

for year in {2019..2024}; 
do 
    wget $prefix/services/services-$year.csv.gz -O data/services-$year.csv.gz && \
    gzip -f -d data/services-$year.csv.gz
done