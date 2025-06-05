from datasets import load_dataset

def load_data():
    """
    Load dataset from Hugging Face.
    """
    dataset = load_dataset("nlphuji/flickr30k")['test']
    return dataset