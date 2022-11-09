import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_size = "2B"  # 350M, 2B, 6B, 16B
model_kind = "mono"
model_id = f"Salesforce/codegen-{model_size}-{model_kind}"
dev = 'cuda'


def main():
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(dev)

    inputs = tokenizer(
        "# this function prints hello world",
        return_tensors="pt",
    ).to(dev)
    sample = model.generate(**inputs, max_length=128)
    output = tokenizer.decode(
        sample[0],
        truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"],
    )

    print(output)


if __name__ == '__main__':
    main()
