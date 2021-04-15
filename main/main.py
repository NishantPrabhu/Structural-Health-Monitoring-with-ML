
import os 
import yaml
import argparse 
import models 


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", type=str, required=True, help='Configuration file path')
    ap.add_argument("-m", "--model", type=str, required=True, help='Kind of model to train')
    args = vars(ap.parse_args())

    config = yaml.safe_load(open(args['config'], 'r'))

    if args['model'] == 'lstm':
        trainer = models.RecurrentModel(config['lstm'])
        trainer.train()