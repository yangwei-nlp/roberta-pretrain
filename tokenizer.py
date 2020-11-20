from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(trainer, ["wikitext-2-raw/wiki.train.raw", "wikitext-2-raw/wiki.valid.raw", "wikitext-2-raw/wiki.test.raw"])

output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)

print(2333)

