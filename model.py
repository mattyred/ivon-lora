import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


def get_model_with_lora(model_name_or_path, lora_r, lora_alpha, lora_dropout):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True)

    target_modules = ["v_proj", "q_proj"]
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


class WrappedModel(torch.nn.Module):
    def __init__(self, model, task_name, tokenizer):
        super().__init__()

        if task_name == "boolq":
            self.id_list = [tokenizer.encode("False")[1], tokenizer.encode("True")[1]]
        elif (task_name == "openbookqa") or ("ARC" in task_name):
            self.id_list = [
                tokenizer.encode("A")[1],
                tokenizer.encode("B")[1],
                tokenizer.encode("C")[1],
                tokenizer.encode("D")[1],
            ]
        elif "winogrande" in task_name:
            self.id_list = [tokenizer.encode("A")[1], tokenizer.encode("B")[1]]

        self.model = model

    def forward(self, **kwargs):
        kwargs.pop("labels", None)
        output_dict = self.model(**kwargs)
        logits = output_dict["logits"]
        return logits[:, -1, self.id_list]
