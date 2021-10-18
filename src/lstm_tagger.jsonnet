// jsonnet allows local variables like this
local embedding_dim = 1024;
local hidden_dim = 100;
local num_epochs = 20;
local patience = 5;
local batch_size = 4;
local learning_rate = 0.1;
local dropout = 0.2;
local cuda_device= 0;

{
    "train_data_path": '/home/username/neural-factoid-geoqa/data/cross/train1.json',
    "validation_data_path": '/home/username/neural-factoid-geoqa/data/cross/test1.json',
    "dataset_reader": {
        "type": "geotag-reader",
        "token_indexers": {
            "tokens": {
              "type": "single_id",
              "lowercase_tokens": true
            },
            "elmo": {
              "type": "elmo_characters"
            }
          }
        },
    "model": {
        "type": "lstm-tagger",
        "word_embeddings": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 50,
                    "pretrained_file": "/home/username/neural-factoid-geoqa/embedding/glove/glove.6B.50d.txt",
                    "trainable":false
                },
                "elmo": {
                    "type": "elmo_token_embedder",
                    "options_file": "/home/username/neural-factoid-geoqa/embedding/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
                    "weight_file": "/home/username/neural-factoid-geoqa/embedding/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
                    "do_layer_norm": false,
                    "dropout": 0.5
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": embedding_dim+50,
            "hidden_size": hidden_dim
        },
        "dropout": dropout
    },
    "iterator": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": [["sentence","num_tokens"]]
    },
    "trainer": {
        "num_epochs": num_epochs,
        "cuda_device": cuda_device,
        "optimizer": {
            "type": "sgd",
            "lr": learning_rate
        },
        "patience": patience
    }
}
