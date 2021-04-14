# Structural-Health-Monitoring-with-ML
Code for our project involving structural health monitoring with machine learning.

This code is a work in progress!

## Usage instructions
Assuming git has been installed in your command line, clone this branch of the repository locally:

```
git clone --branch nishant https://github.com/NishantPrabhu/Structural-Health-Monitoring-with-ML
cd Structural-Health-Monitoring-with-ML
```

Install the required packages.

```
pip3 install -r requirements.txt
```

Run `main.py` with following command line arguments.

```
python3 main.py --config 'configs/main.yaml' --model 'lstm'
```

Configuration for the models can be changed in `configs/main.yaml`. The command above will train the model for default number of epochs and save the last model state and best model (based on validation accuracy) under `outputs`. 
