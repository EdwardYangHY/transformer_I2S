model_1=/net/papilio/storage2/yhaoyuan/transformer_I2S/saved_model/I2U/VC_5_captions_224/beam_val_no_uLM_no_sen
model_2=/net/papilio/storage2/yhaoyuan/transformer_I2S/saved_model/I2U/VC_5_captions_224/beam_val_uLM_ungated_no_sen
model_3=/net/papilio/storage2/yhaoyuan/transformer_I2S/saved_model/I2U/VC_5_captions_224/beam_val_uLM_gated_no_sen

python3 test_i2s.py -m $model_1
python3 test_i2s.py -m $model_2
python3 test_i2s.py -m $model_3