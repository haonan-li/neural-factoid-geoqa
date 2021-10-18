from argparse import Namespace
from allen_utils import *
from allennlp.models.archival import load_archive

args = Namespace(
    archive_file = '../tmp/checkpoints',
    cuda_device = 0,
    overrides = '',
    weights_file = None,
    output_file = '../data/result/lstm-tag.json',
    predict_file = '../data/gold_label_test.json'
)


archive = load_archive(args.archive_file, args.cuda_device, args.overrides, args.weights_file)
config = archive.config
#prepare_environment(config)
model = archive.model
model.eval()

validation_dataset_reader_params = config.pop('validation_dataset_reader', None)
if validation_dataset_reader_params is not None:
    dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
else:
    dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))

predictor = LstmPredictor(model=model,dataset_reader=dataset_reader)
predictor.predict_and_save(args.predict_file, args.output_file)
