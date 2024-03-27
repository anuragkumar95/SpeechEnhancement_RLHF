clean_dir=~/speech-datasets/VoiceBank/train/clean
noise_dir=~/NOISEUS/noise
mixture_dir=~/MixtureRanking/mixtures
ranking_dir=~/MixtureRanking/ranking
nisqa_pt=~/NISQA/weights/nisqa.tar
n = 15000

### Generate audio mixtures
python ./generate_audio_mixtures.py --clean_dir $clean_dir --noise_dir $noise_dir --output $mixture_dir -n $n --mix_aud

### Generate NISQA MOS
python ~/NISQA/run_predict.py --mode predict_dir --pretrained_model $nisqa_pt --data_dir $mixture_dir --num_workers 0 --bs 10 --output_dir $ranking_dir

### Generate ranking files
python ./generate_audio_mixtures.py --mixture_dir $mixture_dir --mos_file $ranking_dir/*.csv --save_dir $ranking_dir --generate_ranks