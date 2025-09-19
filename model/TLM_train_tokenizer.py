import sentencepiece as spm


def train_tokenizer(input_txt="E:\\tinyllm\\tiny_stories.txt", model_prefix="sp_tinystory", vocab_size=7000):
    spm.SentencePieceTrainer.Train(
        f"--input={input_txt} --model_prefix={model_prefix} --vocab_size={vocab_size} "
        "--character_coverage=1.0 --model_type=bpe --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3"
    )
    sp = spm.SentencePieceProcessor()
    sp.load(model_prefix + ".model")
    return sp

train_tokenizer(input_txt="E:\\tinyllm\\tiny_stories.txt", model_prefix="sp_tinystory", vocab_size=6000)
