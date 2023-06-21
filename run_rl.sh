# assume you've already preprocessed data
# and store the file in "/net/papilio/storage2/yhaoyuan/transformer_I2S/data/processed"

echo "Make sure to change the save_name and model path before you run."
cd egs/RL_1_3

# ### Original Settings, all info for all caps.
# spoken_backbone=../../saved_model/I2U/origin_5_captions_256_hubert/codec_hubert_baseline
# spoken_backbone=../../saved_model/I2U/origin_5_captions_256_hubert/prefix_resolution_8_tune_image

# ### Komatsu Settings, all info for only one cap. (4 caps)
# spoken_backbone=../../saved_model/I2U/origin_4_captions_256_hubert_sentence/Codec_No_sentence_8*8_tune_img
# spoken_backbone=../../saved_model/I2U/origin_4_captions_256_hubert_sentence/Codec_8_sentence_8*8_tune_img
# spoken_backbone=../../saved_model/I2U/origin_4_captions_256_hubert_sentence/Prefix_No_sentence_8*8_tune_img
# spoken_backbone=../../saved_model/I2U/origin_4_captions_256_hubert_sentence/Prefix_8_sentence_8*8_tune_img
# spoken_backbone=../../saved_model/I2U/origin_4_captions_256_hubert_sentence/Codec_8_Sentence_8*8_tune_img_half_lr

# ### Komatsu Settings, all info for only one cap. (3 caps)
# spoken_backbone=../../saved_model/I2U/origin_3_captions_256_hubert_sentence/Codec_8_sentence_8*8_tune_img
# spoken_backbone=../../saved_model/I2U/origin_3_captions_256_hubert_sentence/Prefix_8_sentence_8*8_tune_img

# ### EMNLP, Komatsu Setting.
spoken_backbone=../../saved_model/I2U/komatsu_4_captions_256_hubert/Codec_baseline_BLEU_12
# spoken_backbone=../../saved_model/I2U/komatsu_4_captions_256_hubert/Codec_baseline_BLEU_12_7*7_no_tune
# spoken_backbone=../../saved_model/I2U/komatsu_4_captions_256_hubert/Codec_baseline_BLEU_11.7_7*7_double_lr
# spoken_backbone=../../saved_model/I2U/komatsu_4_captions_256_hubert/Prefix_baseline_BLEU_12.5
# spoken_backbone=../../saved_model/I2U/komatsu_4_captions_256_hubert_10_percent/Codec_ID1_BLEU_8.8
# spoken_backbone=../../saved_model/I2U/komatsu_4_captions_256_hubert_10_percent/Prefix_ID2_BLEU_9.7
# spoken_backbone=../../saved_model/I2U/komatsu_4_captions_256_hubert_20_percent/Codec_ID1_BLEU_11.3
# spoken_backbone=../../saved_model/I2U/komatsu_4_captions_256_hubert_20_percent/Prefix_ID2_BLEU_11.3

save_name=emnlp_baseline_rl_dim_50

# This is ordinary Enc-Dec Arch.
# python3 train_DDPG.py -n $save_name -s $spoken_backbone
python3 train_DDPG_komatsu.py -n $save_name -s $spoken_backbone
