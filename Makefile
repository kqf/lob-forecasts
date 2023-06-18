all: data
	python forecasts/main.py

data:
	curl -O -L https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip
	mkdir -p data/
	unzip data.zip -d data/
