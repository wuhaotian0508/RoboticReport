# LoRA Style Distillation

This run freezes the pretrained MoConVQ text generator and trains only LoRA updates.
Pseudo token targets are distilled from the pretrained teacher.
When enabled, HumanML3D motion metrics select among multiple teacher samples before cache writing.

- Examples: 200
- Teacher samples per caption: 4
- Teacher categorical sampling: True
- Teacher top-k: 20
- Teacher temperature: 0.8
- HumanML3D motion selection: True
- LoRA modules: 132
- Trainable parameters: 1966080
- Total parameters: 195591680
