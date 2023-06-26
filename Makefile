all: data
	python forecasts/main.py


evaluate: data data/best.pt
	python forecasts/evaluate.py


infer: data data/best.pt
	python forecasts/infer.py


data/best.pt:
	mkdir -p data/
	curl -O -L https://github.com/kqf/lob-forecasts/releases/download/v0.0.2/best.pt
	mv best.pt data/
	curl -O -L https://github.com/kqf/lob-forecasts/releases/download/v0.0.2/scaler.pickle
	mv scaler.pickle data/


data: file_id = <paste your file id here>
data:
	gdown https://drive.google.com/uc?id=$(file_id)
	mkdir -p data/
	unzip EURUSD.zip -d data/


legacy: data-legacy
	python forecasts/legacy.py


data-legacy:
	curl -O -L https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip
	mkdir -p data-legacy/
	unzip data.zip -d data-legacy/
