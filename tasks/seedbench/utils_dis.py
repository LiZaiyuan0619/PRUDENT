"""SEED-Bench image questions only (data_type == "image"; the video questions with their frames come
after them in lmms-lab/SEED-Bench). Prompt and scoring are those of utils.py; max_new_tokens 16."""


def seed_image_docs(dataset):
    return dataset.filter(lambda t: t == "image", input_columns=["data_type"])  # no image decoding
