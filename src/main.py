import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Salesforce/codegen-2B-mono"


def main():
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    inputs = tokenizer(
        "# this function prints hello world",
        return_tensors="pt",
    ).to('cuda')
    sample = model.generate(**inputs, max_length=128)
    print(
        tokenizer.decode(
            sample[0],
            truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"],
        ))


if __name__ == '__main__':
    main()
