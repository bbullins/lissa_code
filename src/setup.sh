#download data
cd ..
if [ ! -d data ]; then
	mkdir data
fi

if [ ! -d output ]; then
	mkdir output
fi

cd data

if [ ! -e mnist.pkl.gz ]; then
	wget http://deeplearning.net/data/mnist/mnist.pkl.gz
fi

cd ../src

if [ ! -e ../data/mnist49data ]; then
	python data_gen.py
fi

python train.py