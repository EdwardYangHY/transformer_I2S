# assume you've already preprocessed data
# and store the file in "/net/papilio/storage2/yhaoyuan/transformer_I2S/data/processed"

echo "make sure to change the config file before you run."
cd egs/I2U

# This is Baseline Arch
python3 train_i2u_ICASSP.py