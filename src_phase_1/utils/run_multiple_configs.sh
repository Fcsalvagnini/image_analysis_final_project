#!/bin/bash
config_folder="../configs"
input_folder="../images_01"

for file in $(ls $config_folder);
do
    if [[ "$file" == *"config"* ]]; then
        output_folder="${file%.*}"
        output_folder="../output_folder/output_$output_folder"
        config_file="$config_folder/$file"
        echo "****************************************************************"
        echo "[INFO] Executing binary to crop fingerprint with $file"
        echo "****************************************************************"
        ../bin/project01 $input_folder $output_folder $config_file
    fi
done