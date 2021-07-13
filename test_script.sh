## This script is for evaluating performacne, the data should have the label
# python test_single.py --dataset_dir_test 'data/val/' --test_file 'val.txt' --evalPerformance True --max_det 10 --model_path "ViTRes50-1024-16-gmlp-im256.pth" --FRR 16 --heads 0 --backbone 'resnet50' --useFeatMap -2 --testMode 'avg' --FREQ 2 --useBest True --centerCrop 0

## This is for generating the validation result file
python test_single.py --dataset_dir_test '../data/val/' --test_file '../val.txt' --max_det 10 --model_path "ViTRes50-16-gmlp-im256-MF.pth" --FRR 16 --heads 0 --backbone 'resnet50' --useFeatMap -2 --testMode 'avg' --FREQ 2 --useBest True --centerCrop 0 --evalPerformance True

python test_single.py --dataset_dir_test '../data/val/' --test_file '../val.txt' --max_det 10 --model_path "ViTRes50-1024-16-gmlp-im256.pth" --FRR 16 --heads 0 --backbone 'resnet50' --useFeatMap -2 --testMode 'avg' --FREQ 2 --useBest True --centerCrop 0 --evalPerformance True

python test_single.py --dataset_dir_test '../data/val/' --test_file '../val.txt' --max_det 10 --model_path "ViTRes50-12-gmlp-im256.pth" --FRR 12 --heads 0 --backbone 'resnet50' --useFeatMap -2 --testMode 'avg' --FREQ 2 --useBest True --centerCrop 0 --evalPerformance True

## This is for generating the submission file
python test_single.py --dataset_dir_test '../data/test/' --test_file '../test.txt' --max_det 10 --model_path "ViTRes50-16-gmlp-im256-MF.pth" --FRR 16 --heads 0 --backbone 'resnet50' --useFeatMap -2 --testMode 'avg' --FREQ 2 --useBest True --centerCrop 0

python test_single.py --dataset_dir_test '../data/test/' --test_file '../test.txt' --max_det 10 --model_path "ViTRes50-1024-16-gmlp-im256.pth" --FRR 16 --heads 0 --backbone 'resnet50' --useFeatMap -2 --testMode 'avg' --FREQ 2 --useBest True --centerCrop 0

python test_single.py --dataset_dir_test '../data/test/' --test_file '../test.txt' --max_det 10 --model_path "ViTRes50-12-gmlp-im256.pth" --FRR 12 --heads 0 --backbone 'resnet50' --useFeatMap -2 --testMode 'avg' --FREQ 2 --useBest True --centerCrop 0