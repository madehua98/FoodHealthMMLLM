#!/bin/sh

cd /media/fast_data/Data
for i in $(seq 0 2000); do
  if [ -d "$i" ]; then  # 检查名为$i的目录是否存在
    echo "Deleting directory $i"
    rm -r "$i"  # 删除目录及其子目录
  else
    echo "Directory $i does not exist, skipping"
  fi
done
