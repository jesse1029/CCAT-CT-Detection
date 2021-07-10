# Convolutional Transformer for CT Scan Classification
- Chih-Chung Hsu [[cchsu@gs.ncku.edu.tw](mailto:cchsu@gs.ncku.edu.tw "cchsu@gs.ncku.edu.tw")]
- Website: https://cchsu.info
- Institute of Data Science, National Cheng Kung University

**Requirements**
Please see the reruirements.txt for details.
```python
main requirement:
pytorch >= 1.8
```

**Trained models**
[Link of Dropbox [two models available]](https://www.dropbox.com/t/VfnYWlIK9XtzK367 "Link of Dropbox [two models available]")

**Data Preparation**
'Please put the CT files in the "data" folder. Please prepare a file list for testing, the format can refer to "train.txt", "val.txt", and "test.txt". Please aware of that all file list should have label column. For testing purpose only, just fill 0 or 1 in the label column, the code will skip the label information in inference phase if the evalPerformance option is False.'


**How to Use**
If you want to evaluate the performance (having label information), run the script below:
```python
python test_single.py --dataset_dir_test 'data/val/' --test_file 'val.txt' --evalPerformance True --max_det 10 --model_path "ViTRes50-1024-16-gmlp-im256.pth" --FRR 16 --heads 0 --backbone 'resnet50' --useFeatMap -2 --testMode 'avg' --FREQ 2 --useBest True --centerCrop 0
```

If you want to generate the classification result only with csv files, run the script below:
```python
python test_single.py --dataset_dir_test 'data/val/' --test_file 'val.txt' --max_det 10 --model_path "ViTRes50-16-gmlp-im256-MF.pth" --FRR 16 --heads 0 --backbone 'resnet50' --useFeatMap -2 --testMode 'avg' --FREQ 2 --useBest True --centerCrop 0
```

**Check Results**
Please find your results in "result" folder, where a log file, prediction file (in csv) will be automatically generated. 
