LOCAL LOW LATENCY SPEECH TO SPEECH

# Linux

Start With
```
Sudo apt-get update && Sudo apt-get upgrade
```

# Windows WSL

# Continue Here

1. clone the ripository 
```
git clone https://github.com/Zbrooklyn/Local-Low-Latency-Speech-to-Speech.git
```
2. cd into folder
```
cd Local-Low-Latency-Speech-to-Speech
```

3. Install Reqirements
```
pip install -r requirements.txt
```



3. Run these 4 comands.
```
conda create -n openvoice python=3.9
```
```
conda activate openvoice
```
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
```
pip install -r requirements.txt
```

4. create a folder "checkpoints" 
5. Download checkpoints from HERE: (https://myshell-public-repo-hosting.s3.amazonaws.com/checkpoints_1226.zip)
6. Unzip to Checkpoints (basespeakers + converter) 
7. Install LM Studio (https://lmstudio.ai/) 
8. Download Bloke Dolphin Mistral 7B V2 (https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-AWQ) in LM Studio
9. Setup Local Server in LM Studio (https://youtu.be/IgcBuXFE6QE)
10. Start Server
11. Get a reference voice in PATH / PATHS (mp3)
12. RUN talk.py or voice69.py
```
python .\talk.py
```
0r
```
python .\voice69.py
```

# Using LM Studio
7. Install LM Studio (https://lmstudio.ai/) 
8. Download Bloke Dolphin Mistral 7B V2 (https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-AWQ) in LM Studio
9. Setup Local Server in LM Studio (https://youtu.be/IgcBuXFE6QE)

# Using Ollama
install curl
```
sudo apt-get install curl
```

install Ollama
```
curl https://ollama.ai/install.sh | sh
```

Start Server
```
ollama serve
```

Test Ollama
```
ollama run mistral
```
