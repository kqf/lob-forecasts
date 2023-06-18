all: data
	python forecasts/main.py

data: file_id = <paste your file id here>
data:
	gdown https://drive.google.com/uc?id=$(file_id)
	mkdir -p data/
	unzip EURUSD.zip
	mv EURUSD/*.csv data/

# data:
# 	curl -O -L https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip
# 	mkdir -p data/
# 	unzip data.zip -d data/
