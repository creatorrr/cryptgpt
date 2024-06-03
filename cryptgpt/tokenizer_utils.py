from transformers import PreTrainedTokenizerFast

class CryptGPTTokenizer(PreTrainedTokenizerFast):
    @staticmethod
    def clean_up_tokenization(out_string):
        return out_string.replace(' ', "")
