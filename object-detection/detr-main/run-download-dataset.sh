# Make sure folder ./datasets/data exist
mkdir -p ./datasets/data
cd datasets/data

# Download the dataset
curl -O http://images.cocodataset.org/zips/train2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip the dataset
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip