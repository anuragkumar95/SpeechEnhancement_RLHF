```python train.py -r /users/PAS2301/kumar1109/speech-datasets/VoiceBank -o ~/ActorCritic --batchsize 1 --cut_len 32000 --epochs 10 --win_len 25 --gamma 0.99 --tau 0.99```

## TODO
- Frame level input may not formulate an MDP but instead a POMDP as we are partially looking at the current state

- May want to predict the entire spectrogram at once. May want to formulate epoch and episode accordingly. 
