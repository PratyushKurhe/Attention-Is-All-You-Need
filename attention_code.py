
# Install the transformers library if you haven't already
# pip install transformers
import torch
from transformers import MarianMTModel, MarianTokenizer
import sentencepiece
def translate_text(input_text, source_lang="en", target_lang="fr"):
    # Load the pre-trained model and tokenizer
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Tokenize the input text
    input_text = input_text.strip()
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Perform translation
    with torch.no_grad():
        output = model.generate(input_ids, max_length=150, num_return_sequences=1)

    # Decode the translation
    translation = tokenizer.decode(output[0], skip_special_tokens=True)

    return translation

# Example usage:
source_text = "Hello, how are you?"
translated_text = translate_text(source_text, source_lang="en", target_lang="hi")
print("Source Text:", source_text)
print("Translated Text:", translated_text)


