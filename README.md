# A neural POS tagger
Implementation of a neural model for the Part-Of-Speech tagging task for English language.

## Table of Contents
* [General Information](#General-info)
* [Technologies](#Technologies)
* [Setup](#Setup)

## General Information
Part-Of-Speech tagging is a NLP task for labeling words in a text with proper Part-Of-Speech tags. This task is useful for analyzing linguistic properties of a text, but it can be used for improving results on other tasks (e.g. sentiment analysis) as well. Example (from the Speech and Language Processing book, ch.8):
> There/PRO/EX are/VERB/VBP 70/NUM/CD children/NOUN/NNSthere/ADV/RB ./PUNC/

In this project, a neural model is trained on sample.conll data to perform the task.

## Technologies
For the project development, following technologies are used:
- Python: 3.8

## Setup

### Create environment
- Step 0) Clone the repo: `git clone https://github.com/pmarkovic/pos_tagger.git`
- Step 1) Create conda environment: `conda env create -f pavle_rricha.yml`

### Data Preprocessing
- Step 0) Enter into the directory of the project from the terminal: `cd pos_tagger`.   
- Step 1) Generate `sample.conll` file from the given `.conll` files using the command `cat ./data/given_files/*.gold_conll > data/sample.conll` from the terminal.  
- Step 2) For preprocessing, simply run the command `python3 data_preprocess.py --input_file=data/sample.conll --output_dir=data/` or `data_preprocess.sh` file. 
