import torch
import transformers

class Parameters:

    def __init__(self):

        self.max_length_input = 1024
        self.max_length_target = 128

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.checkpoint = 'google-t5/t5-small'
        self.tokenizer: transformers.PreTrainedTokenizerFast = (
            transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.checkpoint))
