#!/usr/bin/env bash
python main.py --model LSTNet --data data/exchange_rate.txt --save save/exchange_rate.pt --hidCNN 50 --hidRNN 50 --L1Loss False --output_fun None
