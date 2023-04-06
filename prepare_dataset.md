## Prepare datasets

This code is used to load up the RefCOCO dataset for evaluation. It was taken from [here](https://github.com/DerrickWang005/CRIS.pytorch).

### 1. COCO 2014

The data could be found at [here](https://cocodataset.org/#download). Please run the following commands to download.

```shell
# download
mkdir datasets && cd datasets
wget http://images.cocodataset.org/zips/train2014.zip

# unzip
unzip train2014.zip -d images/ && rm train2014.zip

```

### 2. Ref-COCO

The data could be found at [here](https://github.com/lichengunc/refer). Please run the following commands to download and convert.

```shell
# download
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip

# unzip
unzip refcoco.zip && rm refcoco.zip

# convert
python ../tools/data_process.py --data_root . --output_dir . --dataset refcoco --split unc --generate_mask

# lmdb
python ../tools/folder2lmdb.py -j anns/refcoco/train.json -i images/train2014/ -m masks/refcoco -o lmdb/refcoco
python ../tools/folder2lmdb.py -j anns/refcoco/val.json -i images/train2014/ -m masks/refcoco -o lmdb/refcoco
python ../tools/folder2lmdb.py -j anns/refcoco/testA.json -i images/train2014/ -m masks/refcoco -o lmdb/refcoco
python ../tools/folder2lmdb.py -j anns/refcoco/testB.json -i images/train2014/ -m masks/refcoco -o lmdb/refcoco

# clean
rm -r refcoco

```
