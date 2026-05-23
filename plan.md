我们在做机器人导论大作业，作业有三个阶段，一是proposal，2是midterm report，三是最终交付。
作业的整体proposal要求：
Intelligent Robot Project Proposal
April 2026
1 Requirements of Project Proposal
For your final project, you are expected to select a robot topic and formulate a concrete
problem. Typical choices include the following:
• Select a recent paper from top conferences or journals such as RSS, CoRL, or RAL as
your baseline. Analyzeits limitations, weaknesses, disadvantages, or futuredirections,
and propose your own method to improve or extend it. You are expected to implement
your ideas and verify them through experiments.
• Select a specific setting (for example, robot manipulation, navigation, locomotion, or
any new setting you define). You should identify or collect a suitable dataset, design
your own algorithm, train your model, and evaluate its performance against appropriate
benchmarks.
• Utilize some existing robotics algorithms to develop an interesting application. The
modules of your application do not need to outperform state-of-the-art methods, but you
must integrate them effectively to create a coherent, functional system.
This proposal should be concise, technically clear, and feasible within the project timeline.
You are required to write a proposal for your final project. There are several requirements
for the proposal:
1. The proposal without reference should be no more than two pages, and you should write
your proposal in LaTex with RSS template (here).
2. The format of proposal is close to mini paper, so the content should include the following
parts at least:
1
• Project title. Choose a concise and descriptive title that reflects the core focus of
your work.
• OvervieworAbstract. Provideaconciseparagraphsummarizingtheentire proposal.
This should include the problem you aim to address, the type of project you are
conducting, the key methods or components you plan to use, and the main outcomes
you expect to deliver.
• Motivation and problem statement. Clearly describe your topic and specify whether
it is a research project or an application project. Explain the problem you intend
to investigate and why it is meaningful or interesting. At this stage, high-level
explanations are sufficient; detailed implementations are not required.
• Literature review. Provide a concise overview of the background and relevant prior
work. Summarize the key ideas from the papers you cite and explain how they relate
to your proposal. Ensure that all references are cited properly.
• Technical plan. State your plan for addressing the proposed topic.
For research projects, describe the method or algorithm you aim to develop. Math
ematical formulations and diagrams of your model architecture are preferred.
If you plan to use existing implementations, specify which ones and how they will
be incorporated. Indicate how you expect to improve or modify current methods.
You do not need finalized solutions at this stage, but you should present a coherent
direction for approaching the problem.
For application projects, diagrams of your system architecture and any prelimi
nary demo materials are preferred. List the key features of your application and
explain how they differ from or improve upon existing applications, if relevant.
• Expected outcomes. Describe the results you expect to achieve. These may include
algorithmic performance, system capabilities, evaluation metrics, or a functional
demo, depending on your project type.
• Potential risks and backup plan. Identify possible obstacles, such as technical
challenges, dataset limitations, or model training issues. Propose backup strategies
that will allow you to continue making progress if your original approach becomes
infeasible.
• Timeline. Provideareasonableschedule that outlines major milestones and expected
completion dates.
• References. List all works cited in the proposal using the standard RSS format.
Ensure that the references are complete, correctly formatted, and limited to materials
2
directly relevant to your topic. Every in-text citation must appear in the list, and no
unrelated or unused references should be included.
3. You should not reuse examples or materials taken directly from other courses or on
line sources; make substantive modifications to ensure the work is genuinely your own.
Plagiarism will make you fail in this project.
2 IdeaPool
Below are several potential project directions. You may choose to build upon one of these
examples or design an entirely original solution. We highly encourage creativity; projects
that explore original ideas beyond these provided baselines will generally receive higher
evaluations. If you have any concerns about your project plan or need guidance, please consult
with the course staff before finalizing your proposal.
2.1 Locomotion Sim2Sim
To verify the generalization capacity of pretrained locomotion policy, we sometimes trans
fer the policy to different simulation platform and validate its performance.
Howto finish this project:
• Reproduce beyondmimic in isaac platform.
• Validate policy’s performance based on this paper
• Transfer the policy to mujoco
• Validate policy’s performance on mujoco platform
• Analysis the sim to sim capacity of the policy
Reference paper&code base: https://beyondmimic.github.io/
2.2 Multiple motion tracking
Existing motion tracking methods like beyondmimic can only tracking single motion
pattern in one policy. We try to increase model scale and datalodaer to track multiple motion
patterns in one policy. How to finish this project:
• Reproduce beyondmimic in isaac platform.
3
• Validate policy’s single motion tracking performance based on this paper
• Try to train the policy with multiple motion
• Validate policy’s performance on multiple motion tracking.
Reference paper&code base: https://beyondmimic.github.io/
Suggestions:You could increase the MLP to bigger size and change the dataloader to learn.
Sometimes, you only change the dataloader will have a good performance on multiple motion
tracking.
2.3 General robot pick-up skills learning
Train a robot pick up policy can generalize to different environments.
• Download the robomimic github repo and carefully read the documentation
• Reproduce one of the pick up tasks like pickup cubes from robomimic
• Modify the task object/table/environment appearance for training
• Validate the policy’s performance under different object or environment settings
Reference paper&code base: https://robomimic.github.io/
2.4 Quadruped robot parkour
To verify the agility and reinforcement learning capabilities of legged robots, we can train
a quadruped robot to overcome complex obstacles like hurdles, gaps, and ramps using vision
and proprioception.
Howto finish this project:
• Reproduce a baseline parkour policy in a simulation platform like Isaac Gym.
• Validate the robot’s performance on standard obstacle tracks (e.g., walking over small
blocks).
• Design a new terrain or obstacle type (e.g., increasing the gap width, or adding moving
obstacles).
• Fine-tune or retrain the policy to adapt to the newly designed environment.
4
• Evaluate the success rate and analyze the failure cases (e.g., falling or colliding).
Reference paper&code base: https://extreme-parkour.github.io/
Suggestions: Training a robot to jump over high obstacles from scratch can be very hard.
You can start with simple, low obstacles and use ”curriculum learning” (gradually increasing
the difficulty of the terrain during training) to help the policy converge faster.
2.5 Language-guided motion generation
Instead of traditional object manipulation, this project focuses on the kinematic generation
of character orhumanoidrobotmotionsbasedonnaturallanguageinstructions. Youwillexplore
how discrete motion representations can be controlled via text.
Howto finish this project:
• DownloadtheMoConvQcodebaseandsetuptherequired environment and datasets (e.g.,
HumanML3D).
• Reproduce the baseline model using the provided pre-trained weights to generate motions
from standard text prompts.
• Create a customized subset of text-motion pairs to introduce a new stylistic movement, or
modify the text condition embedding/sampling strategy during inference.
• Fine-tune the pre-trained model on your customized data or test the zero-shot capabilities
of your modified sampling strategy.
• Validate the policy’s performance by visually comparing the generated 3D motions and
evaluating standard metrics (like Frechet Inception Distance for motions).
Reference paper&code base: https://moconvq.github.io/
Suggestions: Do not train the motion quantization and generation models from scratch,
as it requires massive computational resources. Use the provided pre-trained weights and focus
on the fine-tuning process or modifying the generation conditions. Reserve full training from
scratch only for cases where the target performance cannot be reached otherwise.
2.6 Vision-based robot navigation
Navigate an indoor environment using only first-person visual observations to reach a
specific target.
Howto finish this project:
5
• Install the required visual navigation simulator and clone the ObjectReact repository.
• Reproduce the baseline ObjectGoal navigation policy using the provided pre-trained
weights.
• Modify the environment’s reward function (e.g., add a new penalty for getting too close
to non-target objects to improve safety) OR introduce visual sensor noise (e.g., simulate
imperfect object detection bounding boxes).
• Resume training or fine-tune the agent with your specific modifications.
• Evaluate the agent’s Success rate and Success weighted by Path Length (SPL), analyzing
how your modifications impacted its navigation strategy and robustness.
Reference paper&code base: https://object-react.github.io/
Suggestions: This repository provides a solid starting point for visual navigation. You
can complete a large portion of this project by understanding the provided codebase, tweaking
the reward calculation logic, or modifying the observation inputs without needing to implement
complex reinforcement learning algorithms from scratch.
6，然后我们第一阶段写的proposal：Extending Language-Conditioned Motion
Generation via Style-Aware Fine-Tuning of
MoConVQ
Haotian Wu1,*, Yuhua Luo1,*, Xuyang Yuan1,*
1School of Artificial Intelligence, Shanghai Jiao Tong University
*These authors contributed equally to this work
Abstract—Text-driven 3D human motion generation enables
intuitive control of animated characters and humanoid robots.
MoConVQ [1] demonstrates strong performance in physics
based motion generation with discrete motion representations.
However, like many text-to-motion systems trained on general
purpose corpora, it may be less effective for prompts requiring
fine-grained stylistic control that is underrepresented in the
training data. We propose to adapt MoConVQ by fine-tuning only
its language-conditioning module on a curated stylistic subset
of HumanML3D [2], while keeping the motion representation
backbone frozen. This parameter-efficient strategy is designed to
reduce computational cost and make training feasible on limited
hardware. We evaluate the approach using Fr´ echet Inception
Distance (FID), R-Precision, and qualitative visual comparisons
against the original model.
I. MOTIVATION AND PROBLEM STATEMENT
This is a research project. Natural language is the most
accessible interface for specifying human motions, with appli
cations in game animation, film production, and human–robot
interaction. While MoConVQ represents the state of the art in
physics-based text-to-motion generation, its discrete codebook
and language-conditioning module are trained on general
motion data. As a result, the model performs poorly on
stylistically distinctive prompts (e.g., “a slow, deliberate tai
chi step” or “a graceful folk-dance sway”), which lie outside
its training distribution.
MDM[3]applies a diffusion process directly on raw motion
sequences (rather than in a discrete latent space) conditioned
on text, serving as a representative non-VQ baseline.
T2M-GPT [4] combines a VQ-VAE with a GPT-style
autoregressive transformer over motion tokens, showing that
discrete representations can achieve competitive performance
on text-to-motion benchmarks.
III. TECHNICAL PLAN
Architecture and fine-tuning scope. Figure 1 shows the
inference pipeline. The VQ-VAE encoder/decoder, codebook,
and CLIP text encoder are frozen during fine-tuning. Only
the language-conditioning transformer (mapping CLIP embed
dings to discrete motion codes) is updated. This restricted fine
tuning scope minimizes the number of trainable parameters,
enabling stable fine-tuning on a single RTX 4060 (8 GB
VRAM) with mixed precision.
Text
Prompt
CLIP
Encoder
Lang.
Transformer
(frozen)
Codebook
(frozen)
Problem: Given a natural-language prompt describing a
motion with a particular style, generate a 3D human motion
sequence that captures both the semantic action and the
stylistic nuance. The challenge is adapting a pre-trained large
model to a new style domain with limited compute and without
catastrophic forgetting.
(fine-tune)
codes
VQ-VAE
Decoder
(frozen)
3D
Motion
Fig. 1. Pipeline overview. The VQ-VAE encoder/decoder, codebook, and
CLIP text encoder are frozen during fine-tuning. Only the language
conditioning transformer is updated.
Step 1 — Baseline reproduction. Load the official Mo
ConVQ pre-trained weights and run inference on the Hu
manML3D test set using the original evaluation protocol. We
will run several random seeds and report mean ± std.
II. LITERATURE REVIEW
MoConVQ [1] learns a discrete codebook of motion prim
itives via a residual VQ-VAE and trains a physics-based
tracking controller to follow sampled codes.A text-conditioned
transformer then generates code sequences from natural
language prompts. This two-stage design separates motion
representation learning from text-conditioned control, making
it a suitable backbone for our project.
HumanML3D[2]provides 14,616 motion sequences paired
with 44,970 natural-language descriptions and is widely used
for evaluation with metrics such as FID and R-Precision.
Step 2 — Dataset curation. We first keep the official
HumanML3D train/val/test split unchanged, then filter style
related samples within each split to avoid data leakage. Style
labels are built from a keyword lexicon (e.g., martial, graceful,
folk, deliberate, energetic, fluid) plus a manual audit. We ex
pect 300–600 style-focused clips, with style-specific validation
and test subsets drawn from official val/test.
Step 3 — Fine-tuning. We optimize only the language
conditioning transformer using AdamW. To reduce catas
trophic forgetting, each batch mixes style-focused and general
domain samples. Validation FID and R-Precision are moni
tored jointly every epoch with early stopping.
Step 4 — Evaluation. Compute FID and R-Precision
on the full HumanML3D test set to measure retention of
general capability and the style-specific test split to measure
targeted improvement. All metrics are reported as mean ±
std over several seeds. We also render side-by-side demos for
qualitative comparison.
IV. EXPECTED OUTCOMES
We anticipate delivering both quantitative and qualitative
improvements over the baseline model. Specifically, we expect
the following concrete results:
• Baseline Reproduction: A verified reproduction of the
MoConVQ baseline, achieving a Fr´ echet Inception Dis
tance (FID) within a 0.5 margin of the originally reported
values on the HumanML3D test set.
• Style-Enhanced Fidelity: A fine-tuned language
transformer model that demonstrates at least a 10%
relative improvement in FID on our curated style-specific
evaluation subset.
• Preservation of Generalization: Stable performance on
the full HumanML3D test set (e.g., R-Precision degra
dation of less than 5%), confirming that the fine-tuning
strategy avoided catastrophic forgetting.
• Comprehensive Deliverables: A well-documented code
base for our training pipeline, alongside side-by-side
demo videos comparing the baseline and our fine-tuned
outputs across diverse stylistic prompts.
V. POTENTIAL RISKS AND BACKUP PLAN
• Hardware Constraints (VRAM Limit): Risk: Fine
tuning the language transformer might exceed our 8 GB
VRAM capacity. Backup: We will implement gradient
checkpointing, reduce the batch size to 8 (using gradi
ent accumulation to maintain effective batch size), and
strictly utilize fp16 mixed-precision.
• Catastrophic Forgetting: Risk: The model may over
fit to the style subset and lose broad motion quality.
Backup: We will employ early stopping based on a
joint validation metric (evaluating both style and general
subsets). If full fine-tuning is too aggressive, we will
freeze additional transformer layers or introduce Low
Rank Adaptation (LoRA).
• Data Scarcity for Specific Styles: Risk: Keyword fil
tering on HumanML3D might yield insufficient clips to
meaningfully shift the model’s distribution. Backup: If
the subset is too small (< 200 samples), we will expand
the style vocabulary, relax the filtering rules, or manually
re-annotate a broader subset of existing motions to ensure
sufficient training volume.
VI. TIMELINE
Our team of three will execute the project over an 8
week schedule, with tasks distributed to maximize parallel
development:
Week Milestone & Task Allocation
1–2 Setup & Baseline: Environment configuration; replicate MoConVQ
inference (Member 2); literature deep-dive (Members 1 & 3).
3–4 Data Curation: Filter HumanML3D for style keywords; build
validation splits (Member 3); develop training loop (Member 2).
5–6 Experiments: Execute fine-tuning experiments; tune parameters
& monitor forgetting (Member 2); evaluate checkpoints (Members 1 & 3).
7
Analysis & Rendering: Consolidate quantitative results (FID &
R-Precision) (Member 1); render qualitative 3D videos (Member 3).
8
Finalization: Draft final report, analyze failure cases, clean
up codebase, and prepare project submission (All Members).
REFERENCES
[1] H. Yao, Z. Song, Y. Zhou, T. Ao, B. Chen, and L. Liu,
“MoConVQ: Unified Physics-Based Motion Control via
Scalable Discrete Representations,” in Proc. ICLR, 2024.
[2] C. Guo, S. Zou, X. Zuo, S. Wang, W. Ji, X. Li, and
L. Cheng, “Generating Diverse and Natural 3D Human
Motions from Text,” in Proc. CVPR, 2022, pp. 5152–5161.
[3] G. Tevet, S. Raab, B. Gordon, Y. Shafir, D. Cohen-Or,
and A. H. Bermano, “Human Motion Diffusion Model,”
in Proc. ICLR, 2023.
[4] J. Zhang et al., “Generating Human Motion From Tex
tual Descriptions With Discrete Representations,” in Proc.
CVPR, 2023, pp. 14730–14740.。现在我们要做完剩下的全部内容，满足作业所有要求。
我们的分工你可以参考：
下面给你们一个**三人零基础、一周内可落地完成 midterm report 的执行方案**。我按你们现有 proposal 设计，不建议临时换题。

你们当前项目是：**基于 MoConVQ 做 language-conditioned motion generation，通过 style-aware fine-tuning 改善风格化文本动作生成**。proposal 里已经明确：冻结 VQ-VAE、codebook、CLIP text encoder，只微调 language-conditioning transformer，并在 HumanML3D 风格子集上评估 FID、R-Precision 和定性可视化。

课程 proposal 文档中也明确把 **Language-guided motion generation / MoConVQ / HumanML3D / 风格化子集 / fine-tuning 或 sampling strategy** 列为可选方向，所以你们这个方向是符合课程范围的。

---

# 一句话目标

一周内不要追求真正做出 SOTA 改进。你们的目标是：

> **做出一个可信的 midterm：已经跑通 baseline，构建 style subset，完成至少一个初步实验，放出至少一个表/图/曲线，并清楚说明剩余计划。**

midterm 明确要求“项目已经超过 proposal 阶段”，必须展示具体技术进展、实验或图表。你们不能只写“we will do”。要写“we have implemented / we have constructed / preliminary results show”。

---

# 最低可交付成果

一周后你们至少要交出这些东西：

1. **RSS LaTeX midterm report，1–4 页**
2. **一张 pipeline 图**
   可以复用并改进 proposal 里的 Text → CLIP → Transformer → Codebook/VQ-VAE → Motion 图。你们 proposal 已经有这个结构。
3. **一张 style subset statistics 表**
4. **至少一个 baseline motion generation 可视化截图 / GIF / 视频帧**
5. **一个 preliminary experiment 结果**

   * 最好是 training loss curve；
   * 如果 fine-tuning 跑不动，就用 baseline 在 style prompts 上的 qualitative analysis；
   * 如果 evaluation 跑不动，就放 subset 统计 + baseline generated samples + diagnostic table。
6. **References**
   包括 MoConVQ、HumanML3D、MDM、T2M-GPT，这些 proposal 已经引用过。

---

# 三人分工

设三个人分别为 A、B、C。

## A：报告负责人 / 写作负责人

负责：

* RSS LaTeX 模板；
* midterm report 正文；
* related work；
* references；
* 把 B、C 的结果整理成表格和图；
* 最终合并 PDF。

A 不需要深度写代码，但要保证报告结构和叙事。

---

## B：代码负责人 / baseline 负责人

负责：

* 下载 MoConVQ codebase；
* 配置环境；
* 跑通 pretrained inference；
* 保存 motion 结果；
* 尝试最小 fine-tuning；
* 记录 loss 或运行日志。

B 是最关键的人。

---

## C：数据负责人 / 可视化负责人

负责：

* HumanML3D 文本描述读取；
* style keyword 筛选；
* 统计 style subset；
* 做图表；
* 生成 prompt list；
* 整理 baseline vs fine-tuned 的截图或视频帧。

C 的任务最适合零基础快速产出中期结果。

---

# 一周执行计划

下面按 **Day 1 到 Day 7** 给你们排。严格照着做。

---

# Day 1：确定 midterm 的“保底路线”

## 总目标

今天不要急着训练。先把项目压缩成一个一周内能完成的版本。

你们原 proposal 目标是：

> fine-tune only the language-conditioning transformer on a curated stylistic subset of HumanML3D, while freezing the motion backbone.

中期可以缩小为：

> We have reproduced the MoConVQ inference pipeline, constructed a style-focused subset from HumanML3D, and started preliminary parameter-efficient adaptation experiments.

这句话就是你们 midterm 的核心叙事。

---

## A 今天做什么

### 1. 建立 Overleaf / 本地 LaTeX 工程

用 RSS template。报告结构先写好：

```latex
\title{Style-Aware Adaptation for Language-Conditioned Humanoid Motion Generation}

\begin{abstract}
...
\end{abstract}

\section{Introduction}
\section{Problem Statement and Motivation}
\section{Related Work}
\section{Technical Approach}
\section{Preliminary Experiments and Results}
\section{Remaining Plan}
\section{Conclusion}
\bibliographystyle{IEEEtran}
\bibliography{references}
```

虽然 midterm 要求列了 7 个部分，但你们可以用论文式 section 名称，只要内容覆盖即可。

---

### 2. 写一版空壳

先写每节 3–5 句，不需要完美。重点是留出图表位置：

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/pipeline.pdf}
\caption{Overview of our style-aware adaptation pipeline.}
\label{fig:pipeline}
\end{figure}
```

```latex
\begin{table}[t]
\centering
\caption{Statistics of the curated style-focused subset.}
\label{tab:style_subset}
\begin{tabular}{lccc}
\toprule
Style group & Train & Val & Test \\
\midrule
Martial / slow & -- & -- & -- \\
Dance / graceful & -- & -- & -- \\
Energetic / dynamic & -- & -- & -- \\
Total & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
```

---

## B 今天做什么

### 1. 下载 MoConVQ

进入课程 idea pool 提供的 MoConVQ 页面。课程文档明确建议本方向使用 MoConVQ codebase、HumanML3D、pretrained weights，不建议从头训练。

B 要完成：

* clone repo；
* 阅读 README；
* 记录依赖；
* 找到 pretrained weights 下载方式；
* 找到 inference 命令；
* 找到 HumanML3D 数据准备说明。

今天不要求跑通，但必须建立环境。

---

## C 今天做什么

### 1. 定 style keyword list

你们 proposal 里已经提出了关键词，例如：

* martial
* graceful
* folk
* deliberate
* energetic
* fluid

这些可以直接作为初版。

建议扩展成 5 类：

| Style group            | Keywords                                                       |
| ---------------------- | -------------------------------------------------------------- |
| Martial / slow control | martial, kungfu, tai chi, taichi, slow, deliberate, controlled |
| Graceful / smooth      | graceful, elegant, smooth, fluid, soft, gentle                 |
| Dance / expressive     | dance, dancing, ballet, folk, sway, rhythm                     |
| Energetic / dynamic    | energetic, fast, quickly, jump, jumping, run, leap             |
| Awkward / heavy        | awkward, heavy, limp, stagger, stumble, tired                  |

今天先不筛数据，先确定列表。

---

## Day 1 结束检查

今晚必须有：

* A：LaTeX 空壳；
* B：MoConVQ repo 和环境安装记录；
* C：style keyword list；
* 全组：确定不换题。

---

# Day 2：跑通 baseline inference + 开始筛数据

## 总目标

今天的核心是：**必须让 MoConVQ 生成至少一个 motion。**

哪怕只生成一个 prompt，都算从 proposal 进入 implementation。

---

## B 今天做什么

### 1. 跑官方 demo

优先跑官方最简单的 demo，不要一开始就改代码。

目标 prompt：

```text
A person walks forward slowly.
A person performs a graceful dance.
A person does a slow martial arts movement.
A person jumps energetically.
A person walks in a tired and heavy style.
```

如果能生成视频 / gif / npy / bvh / motion visualization，全部保存。

文件命名：

```text
outputs/baseline/walk_slow.mp4
outputs/baseline/graceful_dance.mp4
outputs/baseline/martial_slow.mp4
outputs/baseline/energetic_jump.mp4
outputs/baseline/heavy_walk.mp4
```

### 2. 记录所有命令

报告中需要 implementation progress，所以必须记录：

```text
Repo commit:
Python version:
CUDA version:
GPU:
Pretrained checkpoint:
Inference command:
Average inference time:
Output format:
```

这可以直接写进 midterm。

---

## C 今天做什么

### 1. 找 HumanML3D 文本描述文件

目标是找到类似：

```text
texts/
train.txt
val.txt
test.txt
```

或者每个 motion 一个 `.txt` 描述文件。

### 2. 写一个最简单的 keyword filter

逻辑：

```python
style_keywords = {
    "martial_slow": ["martial", "kungfu", "kung fu", "tai chi", "taichi", "slow", "deliberate", "controlled"],
    "graceful_smooth": ["graceful", "elegant", "smooth", "fluid", "soft", "gentle"],
    "dance_expressive": ["dance", "dancing", "ballet", "folk", "sway", "rhythm"],
    "energetic_dynamic": ["energetic", "fast", "quickly", "jump", "jumping", "leap"],
    "heavy_awkward": ["awkward", "heavy", "limp", "stagger", "stumble", "tired"]
}
```

输出：

```text
style_subset_train.csv
style_subset_val.csv
style_subset_test.csv
style_subset_statistics.csv
```

每行至少包含：

```text
motion_id, split, text, matched_style_group, matched_keyword
```

---

## A 今天做什么

### 1. 写 Related Work 初稿

只写 4 段，每段 3–4 句。

内容：

1. MoConVQ：baseline，discrete motion representation，physics-based control；
2. HumanML3D：text-motion dataset，标准评价指标；
3. MDM：diffusion text-to-motion；
4. T2M-GPT：VQ + GPT motion token generation。

proposal 里已经有这些内容，可以改写，但不要直接复制。

---

## Day 2 结束检查

今晚必须有：

* B：至少 1 个 baseline 生成结果；
* C：能筛出 style subset 的脚本或初版统计；
* A：Related Work 初稿。

---

# Day 3：构建中期最关键的表和图

## 总目标

今天必须产出 midterm 需要的第一批图表。

midterm 要求至少一个 figure/table/curve。你们至少要有：

1. pipeline 图；
2. style subset statistics 表；
3. baseline generated motion 截图。

---

## C 今天做什么

### 1. 完成 style subset statistics

生成如下表格：

| Style group            | Train | Val | Test | Total |
| ---------------------- | ----: | --: | ---: | ----: |
| Martial / slow control |       |     |      |       |
| Graceful / smooth      |       |     |      |       |
| Dance / expressive     |       |     |      |       |
| Energetic / dynamic    |       |     |      |       |
| Heavy / awkward        |       |     |      |       |
| Total                  |       |     |      |       |

如果数据少，不要慌。可以合并类别：

| Style group       | Keywords                              | #Samples |
| ----------------- | ------------------------------------- | -------: |
| Controlled motion | slow, deliberate, controlled, martial |          |
| Expressive motion | graceful, dance, fluid, elegant       |          |
| Dynamic motion    | energetic, fast, jump, leap           |          |

中期报告里可以解释：

> Some fine-grained style categories contain few samples, so we merge them into broader groups to maintain sufficient training data.

这是合理的。

---

### 2. 做 dataset figure

做一个柱状图：

* x-axis：style group；
* y-axis：sample count；
* title：Distribution of style-focused HumanML3D subset。

这张图就能满足 midterm 的 figure 要求之一。

---

## B 今天做什么

### 1. 批量生成 baseline samples

至少跑 10 个 prompts。

分两类：

#### General prompts

```text
A person walks forward.
A person turns around.
A person sits down.
A person jumps forward.
A person runs in a circle.
```

#### Style prompts

```text
A person walks forward slowly and deliberately.
A person performs a graceful dance motion.
A person moves like doing tai chi.
A person jumps energetically.
A person walks in a tired and heavy style.
```

保存输出。

### 2. 做 baseline qualitative table

让 C 或 A 帮忙整理成表：

| Prompt                                          | Baseline observation                                                     |
| ----------------------------------------------- | ------------------------------------------------------------------------ |
| A person walks forward slowly and deliberately. | Motion follows walking action but style appears weak / generic.          |
| A person performs a graceful dance motion.      | Generated sequence contains turning motion but lacks clear dance rhythm. |
| A person moves like doing tai chi.              | Motion is slow but does not show distinctive martial style.              |

注意：不要夸大。只写你们实际观察到的结果。

---

## A 今天做什么

### 1. 画 pipeline 图

推荐用 draw.io / PowerPoint / LaTeX TikZ 都可以。

内容：

```text
Text Prompt
   ↓
CLIP Text Encoder Frozen
   ↓
Language-Conditioning Transformer Trainable
   ↓
Discrete Motion Codes
   ↓
Frozen VQ-VAE Decoder / Motion Codebook
   ↓
3D Humanoid Motion
```

旁边加一句：

```text
Only the language-conditioning transformer is updated.
```

这和 proposal 技术路线一致。

---

## Day 3 结束检查

今晚必须有：

* style subset 表；
* style subset 分布图；
* pipeline 图；
* baseline samples；
* baseline qualitative observation table。

如果到这里完成了，你们 midterm 已经有保底可交版本。

---

# Day 4：尝试最小 fine-tuning

## 总目标

今天开始尝试训练，但不要把整个项目赌在训练成功上。

你们 proposal 说计划用 AdamW，只优化 language-conditioning transformer，混合 style-focused 和 general-domain samples，监控 FID 和 R-Precision。
中期可以先做简化版本：

> 只跑少量 iterations，记录 training loss 和显存占用。

---

## B 今天做什么

### 1. 找到训练入口

先定位代码里：

* language transformer；
* checkpoint loading；
* dataset loader；
* loss function；
* optimizer。

### 2. 冻结模块

原则：

```python
for p in clip_encoder.parameters():
    p.requires_grad = False

for p in vqvae.parameters():
    p.requires_grad = False

for p in codebook.parameters():
    p.requires_grad = False

for p in language_transformer.parameters():
    p.requires_grad = True
```

实际变量名按 repo 改。

### 3. 最小训练设置

优先用小配置：

```text
batch_size = 1 or 2
gradient_accumulation = 4
epochs = 1
max_steps = 200–1000
fp16 = true
learning_rate = 1e-5 or 5e-5
optimizer = AdamW
```

目标不是训好，而是证明 pipeline 能 train。

### 4. 记录 loss

保存：

```text
step, train_loss
0, ...
10, ...
20, ...
...
```

如果能跑 validation，也保存：

```text
epoch, val_loss
```

---

## C 今天做什么

### 1. 准备训练 prompt subset

从 style_subset_train.csv 里挑 50–200 条样本，先做 small subset。

命名：

```text
style_train_small.csv
style_val_small.csv
style_test_small.csv
```

### 2. 做 manual audit

随机抽 30 条，人工看是否真的是 style-related。做一个小表：

| Audited samples | Style-related | Ambiguous | Incorrect |
| --------------: | ------------: | --------: | --------: |
|              30 |               |           |           |

这个表很有用，能体现你们不是机械 keyword matching。

---

## A 今天做什么

### 1. 写 Technical Approach 初稿

建议分成 4 小节：

```latex
\subsection{Baseline Model}
\subsection{Style-Focused Data Curation}
\subsection{Parameter-Efficient Fine-Tuning}
\subsection{Evaluation Protocol}
```

重点写你们的“当前版本”，不是 proposal 的“未来版本”。

---

## Day 4 结束检查

今晚理想状态：

* B：fine-tuning 能开始跑，并输出 loss；
* C：small style subset 准备好；
* A：technical approach 写完初稿。

如果训练失败，也必须记录错误和原因。报告里可以写成 risk/diagnostic。

---

# Day 5：做 preliminary results

## 总目标

今天要把能交的中期结果固定下来。

结果优先级：

1. training loss curve；
2. baseline vs fine-tuned 生成效果；
3. baseline qualitative analysis；
4. style subset statistics；
5. implementation progress table。

---

## 如果 fine-tuning 成功

B 做：

1. 保存 checkpoint；
2. 用同样 5 个 style prompts 生成 fine-tuned motion；
3. 和 baseline 对比；
4. 保存 loss curve。

C 做：

把 baseline 和 fine-tuned 的视频截成图片，做一张 qualitative comparison figure：

```text
Prompt 1:
Baseline frames | Fine-tuned frames

Prompt 2:
Baseline frames | Fine-tuned frames
```

A 写结果分析：

```text
Preliminary fine-tuning reduces the training loss on the style-focused subset. Qualitatively, the generated motions show stronger correspondence to style-related prompts in several cases, although some motions still remain generic or unstable. We will conduct full FID and R-Precision evaluation in the final stage.
```

---

## 如果 fine-tuning 失败

不要崩。改成 **diagnostic midterm**。

你们仍然可以交，但要写得诚实、技术化。

放这些结果：

1. baseline inference 成功；
2. style subset 构建完成；
3. training pipeline implementation 进展；
4. 失败原因分析；
5. backup plan：LoRA / smaller subset / train fewer layers / inference-time sampling modification。

表格可以这样写：

| Component               | Status      | Evidence                                                   |
| ----------------------- | ----------- | ---------------------------------------------------------- |
| MoConVQ environment     | Completed   | Pretrained inference runs successfully                     |
| HumanML3D preprocessing | Completed   | Style subset statistics obtained                           |
| Baseline generation     | Completed   | 10 prompts generated                                       |
| Fine-tuning loop        | In progress | Freezing modules implemented; memory issue under debugging |
| Evaluation script       | In progress | FID/R-Precision setup under testing                        |

这个表对 midterm 很有用，因为它直接展示 implementation progress。

---

## Day 5 结束检查

今晚必须确定最终报告要放哪些图表：

最低配置：

* Fig. 1 pipeline；
* Table 1 implementation progress；
* Table 2 style subset statistics；
* Fig. 2 baseline generated samples 或 loss curve。

---

# Day 6：写完整 midterm report

## 总目标

今天不要继续无限 debug。开始写完整报告。

报告建议控制在 **3 页左右**。4 页上限，不要写爆。

---

# Midterm report 推荐结构

## Title

建议改成：

```text
Style-Aware Adaptation for Language-Conditioned Humanoid Motion Generation
```

比 proposal 原标题更简洁，也更强调 humanoid / robotics。

---

## Abstract

写法：

```text
Language-conditioned humanoid motion generation provides an intuitive interface for controlling animated characters and humanoid robots. Our project studies style-aware adaptation of MoConVQ, a physics-based motion generation framework with discrete motion representations. Since the proposal, we have focused the project scope on parameter-efficient adaptation of the language-conditioning transformer while freezing the motion representation backbone. We have reproduced the pretrained inference pipeline, constructed a style-focused subset from HumanML3D using a keyword-based lexicon with manual inspection, and conducted preliminary experiments on baseline generation and fine-tuning. Initial results show that the baseline can generate semantically valid motions but often fails to express fine-grained stylistic cues. The remaining work will complete stable fine-tuning, evaluate FID and R-Precision on both full and style-specific test sets, and produce side-by-side qualitative comparisons.
```

如果 fine-tuning 成功，把 “conducted preliminary experiments” 改成 “completed preliminary fine-tuning experiments”。

---

## Problem Statement and Motivation

写清楚：

* 输入：natural-language prompt；
* 输出：3D humanoid motion sequence；
* 问题：style nuance 不足；
* 挑战：limited compute + catastrophic forgetting；
* 与 proposal 的变化：scope 更聚焦到 style subset 和 language transformer adaptation。

---

## Related Work

写短一点。最多半页。

内容：

* MoConVQ 是 baseline；
* HumanML3D 是 dataset；
* MDM 是 diffusion baseline；
* T2M-GPT 是 discrete representation baseline；
* 你们的区别是：不是从头训练大模型，而是做 style-aware adaptation。

---

## Technical Approach

必须写成“我们当前怎么做”。

包括：

### Baseline pipeline

```text
Text prompt → CLIP encoder → language transformer → motion code sequence → VQ-VAE decoder → 3D motion.
```

### Style subset construction

写：

```text
We preserve the original train/validation/test split and apply keyword filtering independently within each split to avoid leakage.
```

这个点很重要，因为 proposal 里已经承诺保持官方 split 避免数据泄漏。

### Fine-tuning

写：

```text
Only the language-conditioning transformer is trainable. The CLIP encoder, VQ-VAE decoder, and motion codebook are frozen.
```

这是 proposal 核心方法。

可以加公式：

[
\mathcal{L}*{\text{train}} =
\mathcal{L}*{\text{CE}}(\hat{z}*{1:T}, z*{1:T}),
]

其中 (z_{1:T}) 是 ground-truth motion code sequence，(\hat{z}_{1:T}) 是 transformer 预测的 code sequence。

如果用了 mixed data：

[
\mathcal{D}*{\text{mix}} =
\alpha \mathcal{D}*{\text{style}} + (1-\alpha)\mathcal{D}_{\text{general}}.
]

---

## Preliminary Experiments and Results

这一节一定要具体。

建议包含：

### Experiment setup

写：

```text
Hardware: RTX 4060 8GB
Precision: fp16 mixed precision
Batch size: ...
Optimizer: AdamW
Learning rate: ...
Trainable module: language-conditioning transformer
Frozen modules: CLIP encoder, VQ-VAE, codebook
```

### Results

按你们真实情况填。

如果有 loss：

```text
Figure 2 shows the preliminary training curve on the style-focused subset. The training loss decreases during early iterations, suggesting that the language-conditioning module can adapt to the selected style prompts.
```

如果只有 baseline：

```text
The pretrained baseline produces valid motions for generic prompts, but the generated motions often become generic walking or turning patterns for fine-grained style prompts. This observation supports our original motivation that style-conditioned generation remains under-specified in the baseline model.
```

### Negative result 也可以写

```text
Our initial fine-tuning attempt encountered GPU memory pressure when using the original batch size. We therefore reduced the batch size and enabled mixed precision. This motivated our remaining plan to use gradient accumulation or LoRA-style adaptation.
```

你们 proposal 里本来就写了 8GB VRAM 风险和 fp16、gradient accumulation、LoRA 备用方案。

---

## Remaining Plan

中期要求有 remaining plan。建议写成表格：

| Remaining task     | Planned action                                                                |
| ------------------ | ----------------------------------------------------------------------------- |
| Stable fine-tuning | Use fp16, gradient accumulation, and early stopping                           |
| Full evaluation    | Report FID and R-Precision on full and style-specific test splits             |
| Ablation           | Compare baseline, full transformer fine-tuning, and frozen-layer/LoRA variant |
| Qualitative demo   | Render side-by-side baseline and adapted motions                              |
| Final report       | Analyze success and failure cases                                             |

---

# Day 7：检查、压页、补证据、提交

## 总目标

今天只做收尾，不做大改。

---

## A 最终检查

逐项对照 midterm requirement：

| 要求                              | 是否满足 |
| ------------------------------- | ---- |
| RSS template                    | 必须满足 |
| 1–4 页                           | 必须满足 |
| mini paper style                | 必须满足 |
| title                           | 必须有  |
| abstract / overview             | 必须有  |
| problem statement               | 必须有  |
| related work                    | 必须有  |
| technical approach              | 必须有  |
| preliminary experiments/results | 必须有  |
| 至少一个 figure/table/curve         | 必须有  |
| references                      | 必须有  |
| 清楚说明自己的 contribution            | 必须有  |

---

## B 最终检查

把所有实验文件整理好：

```text
outputs/
  baseline/
  finetune/
  figures/
  logs/
scripts/
  filter_style_subset.py
  run_baseline_prompts.sh
  train_style_adapter.sh
data/
  style_subset_statistics.csv
report/
  midterm.tex
  references.bib
```

即使不提交代码，也要自己整理好，因为 poster/final 还会用。

---

## C 最终检查

确认所有图都能看懂：

* 图中文字不要太小；
* motion 截图要清楚；
* 表格不要空；
* caption 要说明这张图证明什么。

例如 caption 不要写：

```text
Results.
```

要写：

```text
Preliminary baseline generations on style-related prompts. The baseline generates physically plausible motions but often weakly reflects fine-grained stylistic cues such as ``tai chi'' or ``graceful''.
```

---

# 你们报告里最该强调的“贡献”

中期可以写 3 个 contribution：

1. **We reproduced the MoConVQ inference pipeline for language-conditioned humanoid motion generation.**

2. **We constructed a style-focused HumanML3D subset using a keyword lexicon and manual inspection while preserving the official data split.**

3. **We implemented a parameter-efficient adaptation strategy that freezes the motion representation backbone and updates only the language-conditioning transformer.**

这三点既符合 proposal，又能展示 ownership。

---

# 一周内最现实的技术路线

你们零基础，不建议一开始就硬刚完整 FID 和 R-Precision。建议按三档目标做。

## A 档：理想完成

完成：

* baseline inference；
* style subset；
* fine-tuning 1–3 epoch；
* loss curve；
* baseline vs fine-tuned qualitative comparison；
* partial FID/R-Precision。

这是最强 midterm。

---

## B 档：正常完成

完成：

* baseline inference；
* style subset；
* preliminary fine-tuning loss；
* qualitative observations；
* implementation progress table。

这已经足够像一个合格 midterm。

---

## C 档：保底完成

完成：

* baseline inference；
* style subset statistics；
* generated motion screenshots；
* training pipeline implemented but not stable；
* 写清楚失败原因和 backup plan。

这也能交，但要写得诚实、技术化。

---

# 绝对不要做的事

1. **不要临时换题。**
   你们现在的题目已经符合 idea pool 的 language-guided motion generation 方向。

2. **不要从头训练 MoConVQ。**
   课程文档明确建议不要从头训练 motion quantization 和 generation model，因为计算资源需求很大。

3. **不要把 proposal 原文复制成 midterm。**
   midterm 必须体现“已经做了什么”。

4. **不要虚构指标。**
   如果 FID 没跑出来，就写 qualitative / diagnostic / subset statistics。不要编数字。

5. **不要只写 animation。**
   要强调 humanoid robot / physics-based motion control / language-conditioned humanoid motion generation，否则机器人课程属性会弱。

---

# 最后给你们一个可执行 checklist

按这个清单推进。

## 必做

* [ ] 下载并配置 MoConVQ；
* [ ] 下载或准备 HumanML3D；
* [ ] 跑通 pretrained inference；
* [ ] 生成至少 5 个 baseline motion；
* [ ] 构建 style keyword list；
* [ ] 生成 style subset statistics；
* [ ] 画 pipeline 图；
* [ ] 尝试 fine-tuning；
* [ ] 如果 fine-tuning 成功，画 loss curve；
* [ ] 如果 fine-tuning 失败，写 implementation progress + failure diagnosis；
* [ ] 写 1–4 页 RSS midterm；
* [ ] references 格式正确；
* [ ] 明确写自己的 contribution。

## 推荐图表组合

最稳组合：

1. **Fig. 1：Method pipeline**
2. **Table 1：Style subset statistics**
3. **Table 2：Implementation progress**
4. **Fig. 2：Training loss curve 或 baseline generated motion snapshots**

---

# 结论

你们一周内最稳的打法是：

> **不要追求完整最终结果，先做出“baseline inference + style subset + 初步训练/诊断 + 图表化结果”的 midterm。**

只要报告里能证明你们已经完成了环境搭建、数据筛选、baseline 生成、初步 fine-tuning 或训练诊断，并且有图表支撑，就符合 midterm 的核心要求。
。
你要完整给我们交付两批东西：
第一批是midterm要交的东西，第二批是最后的所有的需要的东西。
根据要求和上面所有内容开始做所有内容。你最好先做一个完整周密的规划然后再开始。