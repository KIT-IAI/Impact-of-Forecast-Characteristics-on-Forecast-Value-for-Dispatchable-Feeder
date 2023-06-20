#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for i in {1..300}
do
  python pipeline.py -id $i
  mkdir  "results_time_parallel/"

#  python pipeline.py -id $i --factor _load2
#  mkdir  "results_load2/"
#  cp -r results/ "results_load2/"
#  rm -rf results
#  python pipeline.py -id $i --factor _ldiv2
#  mkdir  "results_loaddiv2/"
#  cp -r results/ "results_loaddiv2/"
#  rm -rf results
  echo "Finished building $i"
done
