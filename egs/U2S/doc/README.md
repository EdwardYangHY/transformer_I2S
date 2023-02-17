tacotron2
```
rm -rf /net/ageha/storage2/yusuke/anaconda3/envs/tacotron2 
. "/net/ageha/storage2/yusuke/anaconda3/etc/profile.d/conda.sh"
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 
tar xf LJSpeech-1.1.tar.bz2
git clone https://github.com/NVIDIA/tacotron2.git
cd tacotron2
git submodule init; git submodule update
sed -i -- 's,DUMMY,/net/ageha/storage2/yusuke/LAbyRL/U2S/without_docker/LJSpeech-1.1/wavs,g' filelists/*.txt
conda create -n tacotron2 python=3.8
conda activate tacotron2
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
# conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
# tacotron2のモデルダウンロード
python train.py --output_directory=outdir --log_directory=logdir -c ../models/tacotron2_statedict.pt --warm_start
```

tacotron2 python3.7
```
rm -rf /net/ageha/storage2/yusuke/anaconda3/envs/tacotron2py3.7
. "/net/ageha/storage2/yusuke/anaconda3/etc/profile.d/conda.sh"
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 
tar xf LJSpeech-1.1.tar.bz2
git clone https://github.com/NVIDIA/tacotron2.git
cd tacotron2
git submodule init; git submodule update
sed -i -- 's,DUMMY,/net/ageha/storage2/yusuke/LAbyRL/U2S/without_docker/LJSpeech-1.1/wavs,g' filelists/*.txt
conda create -n tacotron2py3.7 python=3.7
conda activate tacotron2py3.7
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
# conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
# tacotron2のモデルダウンロード
python train.py --output_directory=outdir --log_directory=logdir -c ../models/tacotron2_statedict.pt --warm_start
```

tacotron2 python3.7 require
```
rm -rf /net/ageha/storage2/yusuke/anaconda3/envs/tacotron2py3.7
. "/net/ageha/storage2/yusuke/anaconda3/etc/profile.d/conda.sh"
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 
tar xf LJSpeech-1.1.tar.bz2
git clone https://github.com/NVIDIA/tacotron2.git
cd tacotron2
git submodule init; git submodule update
sed -i -- 's,DUMMY,/net/ageha/storage2/yusuke/LAbyRL/U2S/without_docker/LJSpeech-1.1/wavs,g' filelists/*.txt
conda create -n tacotron2py3.7 python=3.7
conda activate tacotron2py3.7-req
pip install -r requirements_old.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
# conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
# tacotron2のモデルダウンロード
python train.py --output_directory=outdir --log_directory=logdir -c ../models/tacotron2_statedict.pt --warm_start
```

apex
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
```


waveglow
```
rm -rf /net/ageha/storage2/yusuke/anaconda3/envs/waveglow
git clone git@github.com:NVIDIA/waveglow.git
cd waveglow
git submodule init
git submodule update
conda create -n waveglow python=3.8
conda activate waveglow
pip3 install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
python3 inference.py -f <(ls model/mel_spectrograms/*.pt) -w model/waveglow_256channels_universal_v5.pt -o . --is_fp16 -s 0.6

```