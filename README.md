# steps to reproduce

download datasets:

https://www.kaggle.com/datasets/janzenliu/cifar-10-batches-py

http://yann.lecun.com/exdb/mnist/

https://github.com/fastai/imagenette

run:

python main.py -d {im|cifar|dtd|mnist|stripes|halves|rand} --num_ims 500 --info_subsample 0.3 --ncs_to_check 8

the "info_subsample" argument is to select a subset of the data for faster computation of entropy
