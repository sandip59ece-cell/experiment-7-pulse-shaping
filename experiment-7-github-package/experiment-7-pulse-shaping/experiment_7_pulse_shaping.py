"""
Experiment 7 - Pulse Shaping and the Nyquist Criterion
Digital Communication Laboratory

Run:
    python experiment_7_pulse_shaping.py

Outputs are saved in ./results/
Dependencies: numpy, matplotlib
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

T = 1.0
SPS = 64
SPAN = 10
t = np.arange(-SPAN*T/2, SPAN*T/2 + 1/SPS, 1/SPS)

def rc_pulse(t, alpha, T=1.0):
    """Raised-cosine impulse response with singularities handled."""
    x = t / T
    h = np.sinc(x)
    if alpha == 0:
        return h
    den = 1 - (2*alpha*x)**2
    h = h * np.cos(np.pi*alpha*x) / den

    singular = np.isclose(np.abs(x), 1/(2*alpha), atol=1e-12)
    h[singular] = (np.pi/4) * np.sinc(1/(2*alpha))
    return h

def rrc_pulse(t, alpha, T=1.0):
    """Root-raised-cosine impulse response."""
    x = t / T
    h = np.zeros_like(x, dtype=float)

    if alpha == 0:
        return np.sinc(x)

    for i, xi in enumerate(x):
        if abs(xi) < 1e-12:
            h[i] = 1 + alpha*(4/np.pi - 1)
        elif abs(abs(xi) - 1/(4*alpha)) < 1e-10:
            h[i] = (alpha/np.sqrt(2))*(
                (1 + 2/np.pi)*np.sin(np.pi/(4*alpha)) +
                (1 - 2/np.pi)*np.cos(np.pi/(4*alpha))
            )
        else:
            num = (np.sin(np.pi*(1-alpha)*xi)
                   + 4*alpha*xi*np.cos(np.pi*(1+alpha)*xi))
            den = np.pi*xi*(1-(4*alpha*xi)**2)
            h[i] = num/den
    return h

def save_plot(name):
    plt.tight_layout()
    plt.savefig(RESULTS/name, dpi=180, bbox_inches="tight")
    plt.close()

def frequency_response(h, dt):
    n = 16384
    H = np.fft.fftshift(np.fft.fft(h, n))*dt
    f = np.fft.fftshift(np.fft.fftfreq(n, d=dt))
    mag = np.abs(H)
    mag /= mag.max()
    db = 20*np.log10(np.maximum(mag, 1e-6))
    return f, db

def main():
    # 1. Pulse responses
    plt.figure(figsize=(8,4.5))
    for alpha in [0.25, 0.5, 1.0]:
        plt.plot(t, rc_pulse(t, alpha), label=f"RC α={alpha}")
    plt.plot(t, rrc_pulse(t, 0.5), "--", label="RRC α=0.5")
    plt.xlim(-4,4); plt.xlabel("Time / T"); plt.ylabel("Amplitude")
    plt.title("Experiment 7: Pulse responses")
    plt.grid(alpha=.25); plt.legend(); save_plot("01_pulse_responses.png")

    # 2. Frequency response
    plt.figure(figsize=(8,4.5))
    for alpha in [0.25,0.5,1.0]:
        f, db = frequency_response(rc_pulse(t,alpha),1/SPS)
        plt.plot(f,db,label=f"RC α={alpha}")
    plt.xlim(-1.2,1.2); plt.ylim(-60,3)
    plt.xlabel("Frequency (Hz, Rs=1)"); plt.ylabel("Magnitude (dB)")
    plt.title("RC frequency responses")
    plt.grid(alpha=.25); plt.legend(); save_plot("02_frequency_responses.png")

    # 3. Nyquist zero crossings
    h = rc_pulse(t,0.5)
    plt.figure(figsize=(8,4.5)); plt.plot(t,h)
    for k in range(-4,5): plt.axvline(k,linestyle=":",alpha=.35)
    plt.xlim(-4,4); plt.xlabel("Time / T"); plt.ylabel("h(t)")
    plt.title("Nyquist zero crossings: RC α=0.5")
    plt.grid(alpha=.25); save_plot("03_zero_crossings.png")

    # 4. Symbol impulse train + pulse shaping
    rng=np.random.default_rng(7)
    symbols=rng.choice([-1,1],size=20)
    up=np.zeros(len(symbols)*SPS); up[::SPS]=symbols
    shaped=np.convolve(up,rc_pulse(t,.5),mode="same")
    tt=np.arange(len(shaped))/SPS
    plt.figure(figsize=(8,4.5))
    plt.plot(tt,shaped); plt.scatter(np.arange(len(symbols)),symbols,s=20)
    plt.xlim(0,19); plt.xlabel("Time / T"); plt.ylabel("Amplitude")
    plt.title("Pulse-shaped symbol stream")
    plt.grid(alpha=.25); save_plot("04_pulse_shaped_stream.png")

    # 5. RRC Tx/Rx cascade
    rrc=rrc_pulse(t,.5)
    cascade=np.convolve(rrc,rrc,mode="full")
    tc=np.arange(len(cascade))/SPS-(len(cascade)-1)/(2*SPS)
    plt.figure(figsize=(8,4.5))
    plt.plot(tc,cascade,label="RRC * RRC")
    plt.plot(t,rc_pulse(t,.5),"--",label="RC")
    plt.xlim(-4,4); plt.xlabel("Time / T"); plt.ylabel("Amplitude")
    plt.title("RRC Tx + Rx cascade")
    plt.grid(alpha=.25); plt.legend(); save_plot("05_rrc_cascade.png")

    # 6. Bandwidth vs roll-off
    alphas=np.array([0,.25,.5,.75,1.0])
    B=(1+alphas)/(2*T)
    plt.figure(figsize=(8,4.5))
    plt.plot(alphas,B,marker="o")
    plt.xlabel("Roll-off factor α"); plt.ylabel("Bandwidth B (Hz)")
    plt.title("Bandwidth versus roll-off factor")
    plt.grid(alpha=.25); save_plot("06_bandwidth_vs_rolloff.png")

    # 7. Mandatory validation table
    sample_times=np.arange(-5,6,dtype=float)
    sample_vals=np.interp(sample_times,t,rc_pulse(t,.5))
    sample_vals[np.where(sample_times==0)[0][0]]=1.0
    print("\nMandatory RC validation (alpha=0.5):")
    print("k\t t/T\t\t h(kT)")
    for k,v in zip(sample_times.astype(int),sample_vals):
        print(f"{k:+d}\t {k:+.1f}\t\t {v:+.6f}")
    print("\nTheory: h(0)=1 and h(kT)=0 for all nonzero integer k.")

if __name__ == "__main__":
    main()
