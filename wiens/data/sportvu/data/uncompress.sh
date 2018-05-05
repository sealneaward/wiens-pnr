#!/bin/sh

for file in *.tar.gz
do
  tar -xvzf "$file"
done
