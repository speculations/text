import torch
import transformers

class Parameters:
    """
    Class: Parameters
    """

    def __init__(self):
        """
        Constructor
        """

        self.input_prefix = 'summarize: '

        self.checkpoint = 'google-t5/t5-small'
        self.tokenizer: transformers.PreTrainedTokenizerFast = (
            transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.checkpoint))
