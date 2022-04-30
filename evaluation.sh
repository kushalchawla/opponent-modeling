expt_name="roberta_trial_1"

cv_no=0
this_expt_name="${expt_name}_cv${cv_no}"
checkpoint_dir="$OppModelingStorage/logs/hierarchical/$this_expt_name/hierarchical/version_0/checkpoints/"

tok_path="$OppModelingStorage/misc/pretrained/roberta_common_tokenizer/tokenizer.pt"

TOKENIZERS_PARALLELISM=true python oppmodeling/eval_analysis.py --model hierarchical --datasets combined --batch_size 25 --tokenizer $tok_path --checkpoint_dir $checkpoint_dir --split test --cv_no $cv_no --output_performance --eval_analysis_mode --use_casino_dialogues --use_roberta