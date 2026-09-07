# Experiment 7 — Pulse Shaping and the Nyquist Criterion

Digital Communication Laboratory experiment.

## Topics
- Rectangular, sinc, raised-cosine (RC) and root-raised-cosine (RRC) pulses
- Nyquist zero-crossing criterion
- Roll-off factor and bandwidth trade-off
- Pulse-shaped symbol stream
- RRC transmitter/receiver cascade
- Mandatory integer-symbol-interval validation

## Run
```bash
pip install -r requirements.txt
python experiment_7_pulse_shaping.py
```

The program prints the RC validation table and saves six graphs in `results/`.

## Key equations

Raised cosine:
h_RC(t) = sinc(t/T) * cos(pi*alpha*t/T) / [1 - (2*alpha*t/T)^2]

RRC:
h_RRC(t) = [sin(pi(1-alpha)t/T) + 4*alpha*(t/T)cos(pi(1+alpha)t/T)]
           / [pi*(t/T)*(1-(4*alpha*t/T)^2)]

Nyquist criterion:
h(0)=1,  h(kT)=0 for every non-zero integer k.

Bandwidth:
B = (1 + alpha) * Rs / 2
