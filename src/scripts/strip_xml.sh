#!/bin/bash

dir=$1
for f in ${1}/*.xml
do
    cat $f | grep "</seg>" | sed "s/<[^>]\+>//g" > $f.txt
done

