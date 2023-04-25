# run_i2uのdataprep(音声生成)が終了したら
cd dataprep/U2S

# out_dir=../../model/U2S/outdir_Origin_5_captions
# out_dir=../../model/U2S/VC_5_captions
out_dir=../../model/U2S/outdir_gtts_hubert_scratch
# checkpoint=/net/papilio/storage2/yhaoyuan/LAbyLM/model/U2S/outdir_kimura/Best_checkpoint_158000

#filelists作成
echo "make filelists"
# python3 preprocess_food.py

# egs u2s
cd ../../egs/U2S
#学習
echo "start train"
python3 train.py -o $out_dir -l logdir #-c $checkpoint --warm_start