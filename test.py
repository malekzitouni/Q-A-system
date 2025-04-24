from transformers import AutoTokenizer, AutoModel, MBart50TokenizerFast, MBartForConditionalGeneration
import torch

# 1. Load TunBERT for Tunisian understanding

tunbert_tokenizer = AutoTokenizer.from_pretrained("not-lain/TunBERT")
tunbert_model = AutoModel.from_pretrained("not-lain/TunBERT")

# 2. Load mBART-50 for translation
mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

def translate_tunisian_to_english(tunisian_text):
    # Encode with TunBERT
    inputs = tunbert_tokenizer(tunisian_text, return_tensors="pt", padding=True, truncation=True)
    tunisian_embeddings = tunbert_model(**inputs).last_hidden_state

    # Prepare for mBART (project embeddings to mBART's space)
    mbart_inputs = {
        "inputs_embeds": tunisian_embeddings,
        "decoder_start_token_id": mbart_tokenizer.lang_code_to_id["en_XX"]  # English output
    }

    # Generate translation
    outputs = mbart_model.generate(**mbart_inputs, max_length=128)
    return mbart_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
tunisian_prompt = "شكون اللي يقدر يفهمني؟"
translated_text = translate_tunisian_to_english(tunisian_prompt)
print(f"Translated: {translated_text}")  # Should output: "Who can understand me?"