# T5_LM


T5 Pretrain

~~~
python t5-pretrain.py --input_length 512 --output_length 128 --num_train_epochs 3 --output_dir t5_pretraining --train_batch_size 8 --learning_rate 2e-4 --model t5-small
~~~


T5 Finetuning

~~~
python t5-finetune.py --input_length
~~~
