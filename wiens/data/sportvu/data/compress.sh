#!/bin/sh

for file in *.pkl
do
  tar -czvf ${file%.pkl}.tar.gz "$file"
done
