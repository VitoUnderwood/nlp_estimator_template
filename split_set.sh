#!/bin/bash
FILENAME="$1"
#read -p "file name: " FILENAME
#echo -e "$FILENAME"
sed -n '1,110000p' $FILENAME > train.json
sed -n '110000,113000p' $FILENAME > dev.json
sed -n '113000,116000p' $FILENAME > test.json