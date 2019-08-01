#!/bin/bash


while true
do
    git add .
    git commit -m 'update'
	#git pull origin master
    git push -u origin master
    sleep 30m
done
