<h1 style="text-align: center;">RENT: Reinforcement Learning via Entropy Minimization</h1>

![RENT: Reinforcement Learning via Entropy Minmization - Figure](./media/RENT_method.png)

RENT is an unsupervised method for training reasoning LLMs by minimizing entropy. We demonstrate on a variety of datasets and models that RENT improves model performance without using any ground truth labels!

RENT is featured in our paper *"Maximizing Confidence Alone Improves Reasoning"* ([link](https://arxiv.org/abs/2505.22660))

RENT is built on top of the **[verl](https://github.com/volcengine/verl)** library.

## Updates:

[10/31]: we release a checkpoint RENT-Qwen-7B of Qwen2.5-7B-Instruct trained with RENT on AIME24. The model can be found at [this huggingface link](https://huggingface.co/aippolit/RENT-Qwen-7B)


## Installation:

Please refer to the existing verl quickstart for installation : [verl Installation](https://verl.readthedocs.io/en/latest/start/install.html)

For the specific vllm version we used (along with other packages), see the `requirements.txt` file.

## Example: Run RENT on Qwen2.5-7B-Instruct using the AIME24 dataset

**Prepare AIME data:**

```
python ./examples/data_preprocess/aime.py --local_dir {path_to_your_dataset}
```

**Run Training:**

*Adjust the configuration in `ppo_trainer.yaml` to match your desired training configuration (number of gpus, batch size, etc.). To override this config somewhere else, see ["Creating Custom Configurations"](#Creating-Custom-Configurations)*

```
python -m verl.trainer.main_ppo exps="[grpo, entropy, format, sampleval, aime]" base_model=Qwen/Qwen2.5-7B-Instruct
```

## Running on Custom Datasets

See verl's documentation on how to prepare data and implement custom reward functions
- Data and Reward Preparation
  - [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
  - [Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)

## Creating Custom Configurations

We use an extensible config setup, allowing you to override default configurations for specific tasks/jobs.

To define a custom configuration, create a new yaml file in `verl/trainer/config/exps`. **NOTE**: you MUST include `# @package _global_` at the beginning of the file in order to override other configs.

To use different configuration files, simply add them to the `exps="[...]"` argument to `verl.trainer.main_ppo`. Note: configurations are applied from left-to-right order, so configs to the right will override configs to the left!

## Citation

```bibtex
@article{prabhudesai2025rent,
    title={Maximizing Confidence Alone Improves Reasoning},
    author={Prabhudesai, Mihir and Chen, Lili and Ippoliti, Alex and Fragkiadaki, Katerina and Liu, Hao and Pathak, Deepak},
    journal={arXiv preprint arXiv:2505.22660},
    year={2025}
}
```
