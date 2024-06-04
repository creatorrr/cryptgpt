from transformers import PreTrainedTokenizerFast

class CryptGPTTokenizer(PreTrainedTokenizerFast):
    @staticmethod
    def clean_up_tokenization(out_string):
        return out_string.replace(' ', "")

CryptGPTTokenizer.register_for_auto_class("AutoTokenizer")
