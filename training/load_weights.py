from email.policy import strict
from dart.models.autoencoder import(
    DartHybridQuantizer,
    DARTAutoEncoder,
    DARTAutoEncoderWithDisc
)



vae_path = "/Users/kairoliu/Documents/Dart/hart/tokenizer"
vae = DARTAutoEncoderWithDisc.from_pretrained(vae_path, ignore_mismatched_sizes=True).vae
