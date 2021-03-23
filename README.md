# A neural POS tagger
Implementation of a neural model for the Part-Of-Speech tagging task for English, German and Serbian.

## Table of Contents
* [General Information](#General-info)
* [Technologies](#Technologies)
* [Setup](#Setup)

## General Information
Part-of-Speech labelling is an important task in Natural Language Procressing, since POS-tags provide more linguistic information about tokens and can help in achieving a deeper understanding of a given text. In this work, we employ a few-shot learning approach, in particular, Prototypical Networks, to investigate whether a POS-tag classifier trained on English examples can also be used for POS-tag classification of two other low resource languages - German and Serbian. We utilize multilingual BERT embeddings to learn contextual representation of words across different languages and then train a Prototypical Network to learn a representation of words and their corresponding POS-tags, in metric space. We perform evaluation on a manually curated corpus containing Universal POS tags for 216 German and 264 Serbian words.
The plots and visualizations from our training can be found at (https://wandb.ai/pmarkovic/multi-pos).

## Technologies
For the project development, following technologies are used:
- Python: 3.8
- Numpy
- PyTorch
- HuggingFace transformers
- Weights & Biases

## Setup

### Final Project Directory Structure  
```
.
├── data_preprocessing
│   ├── data_preprocess.py
│   └── data_preprocess.sh
├── embeddings
│   ├── ptag2embedding.pkl
│   └── train_examples
│       ├── examples_10.pkl
│       ├── examples_11.pkl
│       ├── examples_12.pkl
│       ├── examples_1.pkl
│       ├── examples_2.pkl
│       ├── examples_3.pkl
│       ├── examples_4.pkl
│       ├── examples_5.pkl
│       ├── examples_6.pkl
│       ├── examples_7.pkl
│       ├── examples_8.pkl
│       └── examples_9.pkl
├── gen_embeddings.py
├── model.py
├── models
│   ├── model_10_1_100.pt
│   ├── model_10_1_50.pt
│   ├── ...
├── pavle_rricha.yml
├── README.md
├── results
│   ├── de-test.json
│   └── srb_test_set.json
├── test_data
│   ├── de-test.tsv
│   ├── ontonotes-4.0.info
│   ├── ontonotes-4.0.tsv
│   └── srb_test_set.tsv
├── test.py
├── train.py
└── util.py

```

### Create environment
- Step 0) Clone the repo: `git clone https://github.com/pmarkovic/pos_tagger.git`
- Step 1) Create conda environment: `conda env create -f pavle_rricha.yml`

### Data Preprocessing 
- Step 0) Enter into the directory of the project from the terminal: `cd pos_tagger`.     
- Step 1) For preprocessing, simply run the command `python3 data_preprocessing/data_preprocess.py` or `data_preprocessing/data_preprocess.sh` file. 
```
usage: data_preprocess.py [-h] [--data_dir DATA_DIR] [--out_dir OUT_DIR]

optional arguments:
  -h, --help           show this help message and exit
  --data_dir DATA_DIR  use this option to provide a path to data dir (default=test_data/ontonetes-4.0).
  --out_dir OUT_DIR    use this option to provide the output dir (default=test_data/)
```
### Generate training examples
- Step 0) To generate training examples run: `python3 gen_embeddings.py --gen_embed`. It takes a while to run (approx. 1 hour).
- Step 1) Remove ./temp_data (or other specified dir) with intermediate results: `rm -r ./temp_data`
```
usage: gen_embeddings.py [-h] [--train TRAIN] [--test TEST]
                         [--temp_dir TEMP_DIR] [--examples_dir EXAMPLES_DIR]
                         [--ptag_emb PTAG_EMB] [--bert_model BERT_MODEL]
                         [--gen_embed] [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         use this option to provide the path to English POS-
                        tagged data
  --test TEST           use this option to provide the path to SRB or DE POS-
                        tagged data
  --temp_dir TEMP_DIR   directory where to save intermediate train examples
                        (default=./temp_data).
  --examples_dir EXAMPLES_DIR
                        directory where to save final train examples
                        (default=./embeddings/train_examples).
  --ptag_emb PTAG_EMB   use this option to provide the path to pickled POS-tag
                        embeddings (default: "./embeddings")
  --bert_model BERT_MODEL
                        bert model to use for encoding text (default=bert-
                        base-multilingual-cased).
  --gen_embed           flag to indicate if embeddings should be generated
                        (default=False).
  --batch_size BATCH_SIZE
                        batches of sentences will be created with the given
                        size for generating embeddings (default=50)

```

### Train 
-  Run `python3 train.py`
```
usage: train.py [-h] [--model_path MODEL_PATH] [--bert_model BERT_MODEL]
                [--mdim MDIM] [--bdim BDIM] [--seed SEED] [--k K] [--n N]
                [--ep EP] [--lr LR] [--train TRAIN] [--test TEST]
                [--ptag_emb PTAG_EMB]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        path where to save the trained model
                        (default=models/).
  --bert_model BERT_MODEL
                        bert model to use for encoding text (default=bert-
                        base-multilingual-cased).
  --mdim MDIM           dimension of embeddings in a metric space
                        (default=512).
  --bdim BDIM           dimension of bert model embeddings (default=768).
  --seed SEED           seed for reproducibility purpose (default=777).
  --k K                 number of classes/tags per episode (default=10).
  --n N                 number of examples/shots per class per episode
                        (default=5).
  --ep EP               number of episodes per training (default=100).
  --lr LR               learning rate (default=1e-3).
  --train TRAIN         use this option to provide the path to pickled
                        training examples that were generated by
                        gen_embeddings.pydefault: ./embeddings/train_examples
  --test TEST           use this option to provide the path to the SRB or DE
                        POS-tagged data
  --ptag_emb PTAG_EMB   use this option to provide the path to pickled POS-tag
                        embeddings (default:
                        "./embeddings/ptag2embedding.pkl")

```

### Test
- Run `python3 test.py`
```
usage: test.py [-h] [--model MODEL] [--test TEST] [--bert_model BERT_MODEL]
               [--ptag_emb PTAG_EMB] [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         path to the trained model (default=./models/model.pt).
  --test TEST           path to the test set (default=./test_data/de-
                        test.tsv).
  --bert_model BERT_MODEL
                        bert model to use for encoding text (default=bert-
                        base-multilingual-cased).
  --ptag_emb PTAG_EMB   use this option to provide the path to pickled POS-tag
                        embeddings (default: "./embeddings")
  --batch_size BATCH_SIZE
                        batches of sentences will be created with the given
                        size for generating embeddings (default=50)
```


