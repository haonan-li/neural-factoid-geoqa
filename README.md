# Neural Factoid Geospatial Question Answering


## Abstract

Existing question answering systems struggle to answer factoid questions when geospatial information is involved.
This is because most systems cannot accurately detect the geospatial semantic elements from the natural language questions, or capture the semantic relationships between those elements.
In this paper, we propose a geospatial semantic encoding schema and a semantic graph representation which captures the semantic relations and dependencies in geospatial questions.
We demonstrate that our proposed graph representation approach aids in the translation from natural language to a formal, executable expression in a query language.
To decrease the need for people to provide explanatory information as part of their question and make the translation fully automatic, we treat the semantic encoding of the question as a sequential tagging task, and the graph generation of the query as a semantic dependency parsing task.
We apply neural network approaches to automatically encode the geospatial questions into spatial semantic graph representations.
Compared with current template-based approaches, our method generalises to a broader range of questions, including those with complex syntax and semantics.
Our proposed approach achieves better results on GeoData201 than existing methods.

## Requirements

- python 3.6
- [allennlp](https://github.com/allenai/allennlp) For LSTM-based semantic encoder.
- [transformers](https://github.com/huggingface/transformers) For BERT semantic encoder.
- [am-parser](https://github.com/coli-saar/am-parser) For graph generation.
- [rule-based geo-question encoder](https://github.com/haonan-li/place-qa-AGILE19) To replicate rule-based semantic encoding result.

## Data

All data mentioned in the paper is in [`data`](https://github.com/haonan-li/neural-factoid-geoqa/data) directory.

## Semantic Encoding

Rule-based encoder: To replicate the result of rule-based encoder, you need to follow the instruction of [this](https://github.com/haonan-li/place-qa-AGILE19) to download the relevant tools and gazetteer, then run the [simple encoder](https://github.com/haonan-li/place-qa-AGILE19/blob/master/simple_end_to_end.py).

LSTM encoder: We use the allennlp tools, you need to download the [GloVe](https://nlp.stanford.edu/projects/glove/) embedding files and [ELMo](https://allennlp.org/elmo) weights, replace the locations in [jsonnet](https://github.com/haonan-li/neural-factoid-geoqa/src/lstm_tagger.jsonnet) file, and run `bash lstm_tagger_train.sh`.

BERT encoder: We use the [huggingface/transformers](https://github.com/huggingface/transformers) token-level encoder, simply run `bash bert_tagger_train_pred.sh` to use the BERT encoder.

## Graph Generation

Use [am-parser](https://github.com/coli-saar/am-parser) to generate DM graph representation, then use `graph_build_and_eval.ipynb` to generate the graph use the tagging results and DM graph.
