# Awesome DeepSeek-R1
A collection of the recent reproduction papers and projects on DeepSeek-R1


[🎉](https://api-docs.deepseek.com/zh-cn/news/news250120) The **Awesome DeepSeek-R1 Collection** aims to serve as a centralized repository of high-quality reproductions, adaptations, and extensions of the original DeepSeek-R1 model. This collection will include:

- **Reproduction Papers**: Detailed reports on successful reproductions of DeepSeek-R1, highlighting methodologies, results, and lessons learned.
- **Code Repositories**: Open-source implementations of DeepSeek-R1, providing researchers and developers with the tools to experiment and build upon the model.



## DeepSeek-R1-Zero and DeepSeek-R1

Large Language Models (LLMs) have shown remarkable progress in recent years, with a focus on achieving Artificial General Intelligence (AGI). Post-training methods, such as reinforcement learning, have become crucial for enhancing model performance in reasoning tasks. OpenAI's o1 series models have set a benchmark by introducing inference-time scaling through extended Chain-of-Thought (CoT) reasoning. However, effective test-time scaling remains a challenge. This paper explores the potential of pure reinforcement learning (RL) to improve reasoning capabilities without supervised data, aiming to align with human preferences and enhance performance across diverse tasks.



### DeepSeek-R1-Zero: Reinforcement Learning on the Base Model

DeepSeek-R1-Zero is a model trained via large-scale reinforcement learning without any supervised fine-tuning. The authors use DeepSeek-V3-Base as the base model and employ 🚀 **Group Relative Policy Optimization** (GRPO) as the RL framework. The model is trained using a rule-based reward system that includes accuracy rewards and format rewards, ensuring the model adheres to specified instructions.

### DeepSeek-R1: Reinforcement Learning with Cold Start

To address issues like poor readability and language mixing in DeepSeek-R1-Zero, the authors introduce DeepSeek-R1, which incorporates multi-stage training with cold-start data before RL. The pipeline includes four stages:

1. **Cold Start**: Thousands of long CoT data are collected to fine-tune the DeepSeek-V3-Base model, ensuring readability and human-friendly outputs.
2. **Reasoning-oriented Reinforcement Learning**: The fine-tuned model undergoes RL training, focusing on enhancing reasoning capabilities in tasks like coding, mathematics, and science.
3. **Rejection Sampling and Supervised Fine-Tuning**: New SFT data is generated through rejection sampling on the RL checkpoint, combined with supervised data from DeepSeek-V3 in various domains.
4. **Reinforcement Learning for all Scenarios**: A secondary RL stage aligns the model with human preferences, improving helpfulness and harmlessness while refining reasoning capabilities.



## Techniques in DeepSeek-R1

### Reinforcement Learning-Driven Reasoning Capability Enhancement

- 🚀 **Group Relative Policy Optimization (GRPO)** : To save the training costs of reinforcement learning, DeepSeek-R1 adopts GRPO, which abandons the critic model that is usually the same size as the policy model, and estimates the baseline from group scores instead. This method simplifies the training process and improves training efficiency and model performance.
- **Reward Modeling** : A rule-based reward system is used, mainly consisting of two types of rewards. Accuracy rewards evaluate whether the response is correct, such as requiring the model to provide the final answer in a specified format for math problems with deterministic results, enabling reliable rule-based verification of correctness. Format rewards enforce the model to put its thinking process between specific tags.

​	**Reference papers:** 

- PPO：
  - Proximal Policy Optimization Algorithms [[Paper]](https://arxiv.org/abs/1707.06347)
  - Delve into PPO: Implementation Matters for Stable RLHF [[Paper]](https://openreview.net/forum?id=rxEmiOEIFL)
- GRPO：Group Robust Preference Optimization in Reward-free RLHF [[Paper]](https://openreview.net/forum?id=PRAsjrmXXK) [[Code]](https://github.com/rsshyam/GRPO)

### Cold Start Data and Multi-stage Training

- **Cold Start Data** : Unlike DeepSeek-R1-Zero, DeepSeek-R1 uses a small amount of long Chain-of-Thought (CoT) data for fine-tuning the model before reinforcement learning, as the initial execution body of RL. This helps to avoid the unstable cold start stage of the basic model in the early stage of reinforcement learning training.
- **Multi-stage Training** : DeepSeek-R1 employs a multi-stage training strategy. First, it conducts supervised fine-tuning with thousands of high-quality examples, then focuses on reasoning tasks with reinforcement learning. Additionally, new training data is collected through rejection sampling, and finally, reinforcement learning is applied to cover all tasks. This method solves the limitations of DeepSeek-R1-Zero in terms of readability and language consistency, and achieves higher performance.

### Long Chain of Thought (CoT) Reasoning Technology

DeepSeek-R1 uses long CoT reasoning technology, with a thought chain length that can reach tens of thousands of words. This allows the model to break down complex problems step by step and solve them through multi-step logical reasoning, showing higher efficiency in complex tasks.

**Reference papers:** 

- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (CoT) [[Paper]](https://openreview.net/forum?id=_VjQlMeSB_J)
- Tree of Thoughts: Deliberate Problem Solving with Large Language Models (ToT) [[Paper]](https://arxiv.org/abs/2305.10601)
- Demystifying Long Chain-of-Thought Reasoning in LLMs [[Paper]](https://arxiv.org/abs/2502.03373) [[Code]](https://github.com/eddycmu/demystify-long-cot)

### Model Distillation Support

🚀 DeepSeek-R1 supports users in using its output for model distillation to train smaller models, meeting the needs of different application scenarios.

**Reference papers:** 

- Distilling the Knowledge in a Neural Network [[Paper]](https://arxiv.org/abs/1503.02531)
- Knowledge Distillation: A SurveyKnowledge Distillation: A Survey [[Paper]](https://arxiv.org/abs/2006.05525)



### Reproduction Projects and Papers

|                         | Links                                                        | Base Model                                                   | Training Data                                                | Tasks                     | Training Resources                                  |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------- | --------------------------------------------------- |
| DeepSeek-R1             | [[Demo]](https://chat.deepseek.com/)<br />[[Github]](https://github.com/deepseek-ai/DeepSeek-R1)<br />[[Paper]](https://arxiv.org/abs/2501.12948) | DeepSeek-V3                                                  | -                                                            | 🔢 Mathematical  Reasoning | -                                                   |
| Open-R1                 | [[Code]](https://github.com/huggingface/open-r1)<br />[[Demo]](https://open-r1.com/) | Qwen2.5-1.5B-Instruct/ Qwen2.5-Math-7B/ Qwen-32B/Qwen-72B/ Llama-8B/ Llama-70B | [[Bespoke-Stratos-17k]](https://huggingface.co/datasets/HuggingFaceH4/Bespoke-Stratos-17k) | 🔢 Mathematical  Reasoning | 1-2 Nodes of H100s                                  |
| TinyZero                | [[Code]](https://github.com/Jiayi-Pan/TinyZero)<br />[[Experiment Log]](https://wandb.ai/jiayipan/TinyZero) | QWen-2.5-3B Instruct                                         | [[Countdown-Tasks-3to4]](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) | ✏ Countdown               | 4 A800s                                             |
| Mini-R1                 | [[Code]](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb)<br />[[Tutorial]](https://www.philschmid.de/mini-deepseek-r1) | Qwen2.5-3B-Instruct                                          | [[Countdown-Tasks-3to4]](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) | ✏ Countdown               | 4 H100s                                             |
| DeepScaleR              | [[Code]](https://github.com/agentica-project/deepscaler)<br />[[Model]](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview)<br />[[Blog]](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) | DeepSeek-R1-Distill-Qwen-1.5B                                | [DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) /  [JSON Dataset](https://github.com/agentica-project/deepscaler/tree/main/deepscaler/data) | 🔢 Mathematical Reasoning  | 8 A800s for single-node and 32 A800s for multi-node |
| LIMO                    | [[Code]]()<br />[[Model]]()<br />[[Paper]](https://github.com/GAIR-NLP/LIMO) | Qwen2.5-32B-Instruct                                         | [[LIMO]](https://huggingface.co/datasets/GAIR/LIMO)          | 🔢 Methematical Reasoning  | -                                                   |
| Logic-RL                | [[Code]](https://github.com/Unakar/Logic-RL)<br />[[Blog]](https://evxpwrsfkdb.feishu.cn/docx/NokEdaMBmo6aqZxVdxkcSm2cnab) | Qwen2.5-7B-Instruct-1M                                       | [[Synthesis Data]](https://github.com/Unakar/Logic-RL/tree/main/data/kk/instruct) | 📖 Logic Puzzle            | -                                                   |
| s1                      | [[Code]](https://github.com/simplescaling/s1)<br />[[Paper]](https://arxiv.org/abs/2501.19393)<br />[[Model]](https://hf.co/simplescaling/s1-32B) | Qwen2.5-32B-Instruct                                         | [[s1K]](https://hf.co/datasets/simplescaling/s1K)            | 🔢 Methematical Reasoning  | 16 H100s (26min)                                    |
| SimpleRL-reasom         | [[Code]](https://github.com/hkust-nlp/simpleRL-reason)       | Qwen2.5-Math-7B                                              | [[Data]](https://github.com/hkust-nlp/simpleRL-reason)       | 🔢 Methematical Reasoning  | 6 H/A100s (Minimum)                                 |
| oat-zero                | [[Code]](https://github.com/sail-sg/oat-zero)<br />[[Blog]](https://oatllm.notion.site/oat-zero) | Qwen-2.5-3B                                                  | [[Countdown-Tasks-3to4]](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) | ✏ Countdown               | -                                                   |
| TTS (Test time scaling) | [[Blog]](https://ryanliu112.github.io/compute-optimal-tts/)<br />[[Paper]](https://arxiv.org/abs/2502.06703) | Opensource models                                            | -                                                            | 🔢 Mathematical Reasoning  | -                                                   |
| Datawhale-R1            | [[Code]](https://github.com/datawhalechina/unlock-deepseek)<br />[[Blog]](https://datawhalechina.github.io/unlock-deepseek/) | Qwen2.5-3B-Instruct                                          | [[Countdown-Tasks-3to4]](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) | ✏ Countdown               | 3 A800s                                             |
| demystify-long-cot      | [[Code]](https://github.com/eddycmu/demystify-long-cot)<br />[[Paper]](https://arxiv.org/abs/2502.03373) | Llama-3.1-8B/ Qwen2.5-Math-7B                                | MATH and WebIT                                               | 🔢 Mathematical Reasoning  | -                                                   |

## Citation

❤ If you find our repository useful in your research, please star us ⭐ and consider citing:

```
@misc{liu2025DeepSeekR1_Reproduce,
  title={Awesome DeepSeek-R1},
  author={Haoyang Liu and Zhihai Wang and Wenyu Mao},
  year={2025},
  howpublished={\url{https://github.com/haoyangliu123/awesome-deepseek-r1}},
  note={Github Repository},
}
```
