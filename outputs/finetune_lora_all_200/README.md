# LoRA Style Distillation

This run freezes the pretrained MoConVQ text generator and trains only LoRA updates.
Pseudo token targets are distilled from the pretrained teacher, not from ground-truth HumanML3D motion tokens.

- Examples: 200
- LoRA modules: 162
- Trainable parameters: 2396160
- Total parameters: 196021760
