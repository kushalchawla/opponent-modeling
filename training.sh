learning_rate=0.00002
dropout=0.1
loss_dropout=0.15

expt_name="roberta_trial_1"
random_seeds=($(shuf -e -n 5 {1..10000}))

cv_no=0
this_expt_name="${expt_name}_cv${cv_no}"

tok_path="$OppModelingStorage/misc/pretrained/roberta_common_tokenizer/tokenizer.pt"

TOKENIZERS_PARALLELISM=true python oppmodeling/main.py --model hierarchical --datasets combined --gpus 1 --batch_size 25 --max_epochs 20 --log_every_n_steps 1 --val_check_interval 1.0 --tokenizer $tok_path --default_root_dir $OppModelingStorage/logs/hierarchical/$this_expt_name --cv_no $cv_no --overall_seed ${random_seeds[$cv_no]} --ranking_margin 0.3 --learning_rate $learning_rate --dropout $dropout --loss_dropout $loss_dropout --use_roberta --use_casino_dialogues --use_casino_reasons --use_dnd_dialogues