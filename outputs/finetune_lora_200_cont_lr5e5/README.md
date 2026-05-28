# LoRA Style Distillation

This run freezes the pretrained MoConVQ text generator and trains only LoRA updates.
Pseudo token targets are distilled from the pretrained teacher, not from ground-truth HumanML3D motion tokens.

- Examples: 200
- LoRA modules: 132
- Trainable parameters: 1966080
- Total parameters: 195591680
