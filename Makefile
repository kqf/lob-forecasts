all: data
	python forecasts/explore.py


data: file_id = <paste your file id here>
data:
	# gdown https://drive.google.com/uc?id=$(file_id)
	mkdir -p data/
	unzip EURUSD.zip -d data/


legacy: data-legacy
	python forecasts/legacy.py


data-legacy:
	curl -O -L https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip
	mkdir -p data-legacy/
	unzip data.zip -d data-legacy/
