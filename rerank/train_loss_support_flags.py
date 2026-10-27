"""Train common-neutral-loss flag classifiers from train-only features/spectra."""
from rerank.training_auxiliary import train

if __name__ == '__main__':
    train('loss-flags')
