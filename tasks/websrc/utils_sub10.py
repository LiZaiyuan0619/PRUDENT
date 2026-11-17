"""The WebSRC dev subset for websrc_val_sub10.yaml -- every 10th question, in dev order."""


def every_10th(dataset):
    return dataset.select(range(0, len(dataset), 10))
