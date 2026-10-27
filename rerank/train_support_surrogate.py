"""Train the common-neutral-loss support surrogate from train-only features."""
from rerank.training_auxiliary import train

if __name__ == '__main__':
    train('support')
