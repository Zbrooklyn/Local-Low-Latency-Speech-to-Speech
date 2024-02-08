LOCAL LOW LATENCY SPEECH TO SPEECH

1. git clone https://github.com/All-About-AI-YouTube/low-latency-sts.git
2. cd low-latency-sts
3. run:
- conda create -n openvoice python=3.9
- conda activate openvoice
- conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
- pip install -r requirements.txt
4. create a folder "checkpoints"
5. Download checkpoints from HERE: https://myshell-public-repo-hosting.s3.amazonaws.com/checkpoints_1226.zip
6. Unzip to Checkpoints (basespeakers + converter)
7. Install LM Studio (https://lmstudio.ai/)
8. Download Bloke Dolphin Mistral 7B V2 (https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-AWQ) in LM Studio
9. Setup Local Server in LM Studio (https://youtu.be/IgcBuXFE6QE)
10. Start Server
11. Get a reference voice in PATH / PATHS (mp3) - Line 204
12. RUN talk.py or voice69.py
