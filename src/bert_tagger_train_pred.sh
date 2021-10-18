# please download the transfomers first
for i in 0 1 2 3 4; do
python /home/haonanl5/tools/transformers/examples/pytorch/token-classification/run_ner.py \
    --train_file /home/haonanl5/hackfest-2019/data/cross/train${i}.json \
    --validation_file /home/haonanl5/hackfest-2019/data/cross/test${i}.json \
    --test_file /home/haonanl5/hackfest-2019/data/cross/test${i}.json \
    --model_name_or_path bert-base-uncased \
    --output_dir /home/haonanl5/hackfest-2019/data/result/bert/${i} \
    --do_train \
    --do_predict \
    --num_train_epochs 10 \
    --logging_steps 10 \
    --save_steps 10 \
    --overwrite_output_dir
done
