# Visual-Recognition-HW1
## Environment
PyTorch GPU

Python 3.8

Numpy

PIL

## Installation
```
git clone https://github.com/Kelpen/Visual-Recognition-HW1
cd Visual-Recognition-HW1
```

efficientnet_pytorch
```
pip install efficientnet_pytorch
```

Put the "test_data_list.txt" under the same directory with "training_labels.csv"

## Train
```
python Trainer.py data_root=your/path/to/dataset
```
## Test
```
python Test.py data_root=your/path/to/dataset
```
