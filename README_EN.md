<div align="center">
<h1>
  BlueLM
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/vivo-ai" target="_blank">Hugging Face</a> • 👾 <a href="https://www.modelscope.cn/organization/vivo-ai" target="_blank">ModelScope</a> • 🤖 <a href="https://www.wisemodel.cn/organization/vivo-ai" target="_blank">wisemodel</a> • 📜 <a href="MODEL_LICENSE_EN.pdf" target="_blank">LICENSE</a> • 🎯 <a href="https://developers.vivo.com/product/ai/bluelm" target="_blank">vivo Developers</a> • 🗨 <a href="resources/wechat.png" target="_blank">WeChat</a>
</p>

<div align="center">

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="README.md">中文</a>
    <p>
</h4>

</div>

# Table of Contents

- [📔 Models Introduction](#Models-Introduction)
- [📊 Benchmark Results](#Benchmark-Results)
- [🚀 Inference and Deployment](#Inference-and-Deployment)
- [⚒ Fine-tuning the Model](#Fine-tuning-the-Model)
- [📚 Disclaimer, License and Citation](#Disclaimer-License-and-Citation)
- [📠 Contact Us](#Contact-Us)

# Models Introduction

BlueLM is a large-scale open-source language model independently developed by the vivo AI Lab. This release includes 2K and 32K context length versions for both Base and Chat models.

- **High-quality Data**: BlueLM is trained on a high-quality data with 2.6 trillion tokens. Our train corpus mainly consists of Chinese and English data, with a small amount of Japanese and Korean data.
- **Stronger Performance**: BlueLM-7B-Chat achieves a strong competitive performance in C-Eval and CMMLU benchmarks of the same size.
- **Longer Context**: We have extended the context length of both BlueLM-7B-Base-32K and BlueLM-7B-Chat-32K models from 2K to 32K. The models can support longer context understanding while maintaining the same basic capabilities.
- **Model License**: BlueLM weights are open for academic research and commercial use. 

The release versions and hugging face download links are listed in the table below:

|        | Base Model                                                                      | Chat Model                                                                     | 4bits Quantized Chat Model                                                                          |
|:-------|:--------------------------------------------------------------------------------|:-------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|
| 7B-2K  | 🤗 [BlueLM-7B-Base](https://huggingface.co/vivo-ai/BlueLM-7B-Base)              | 🤗 [BlueLM-7B-Chat](https://huggingface.co/vivo-ai/BlueLM-7B-Chat)             | 🤗 [BlueLM-7B-Chat-4bits](https://huggingface.co/vivo-ai/BlueLM-7B-Chat-4bits)                      |
| 7B-32K | 🤗 [BlueLM-7B-Base-32K](https://huggingface.co/vivo-ai/BlueLM-7B-Base-32K)      | 🤗 [BlueLM-7B-Chat-32K](https://huggingface.co/vivo-ai/BlueLM-7B-Chat-32K)     | -                                                                                                   |

Welcome to read our technical report [BlueLM: An Open Multilingual 7B Language Model](https://github.com/vivo-ai-lab/BlueLM/blob/main/BlueLM_technical_report.pdf)!

We will release the 13B language model and the 7B-vl multi-modal language model soon!

# Benchmark Results

To ensure the consistency of model evaluation, we use [OpenCompass](https://opencompass.org.cn/leaderboard-llm) to evaluate the performance on relevant leaderboards. We conducted extensive tests on C-Eval, MMLU, CMMLU, GaoKao, AGIEval, BBH, GSM8K, MATH and HumanEval datasets across general ability, mathematical ability and coding ability.

## Benchmarks

- [C-Eval](https://cevalbenchmark.com/index.html) is the first comprehensive Chinese evaluation suite designed to assess advanced knowledge and reasoning abilities of foundation models in a Chinese context.
- [MMLU](https://arxiv.org/abs/2009.03300) covers 57 tasks including elementary mathematics, US history, computer science, law, and more.
- [CMMLU](https://github.com/haonan-li/CMMLU) is a comprehensive Chinese evaluation benchmark specifically designed to assess the knowledge and reasoning abilities of language models in the context of the Chinese language.
- [Gaokao](https://github.com/OpenLMLab/GAOKAO-Bench) is an evaluation framework that utilizes Chinese high school entrance examination (GAOKAO) questions as a dataset to evaluate the language understanding and logical reasoning abilities of large language models.
- [AGIEval](https://github.com/ruixiangcui/AGIEval) is a human-centric benchmark specifically designed to evaluate the general abilities of foundation models in tasks pertinent to human cognition and problem-solving.
- [BBH](https://huggingface.co/datasets/lukaemon/bbh) is a suite of 23 challenging BIG-Bench tasks.
- [GSM8K](https://github.com/openai/grade-school-math) is a dataset of 8.5K high quality linguistically diverse grade school math word problems created by human problem writers.
- [MATH](https://huggingface.co/datasets/competition_math) is a new dataset of 12,500 challenging competition mathematics problem.
- [HumanEval](https://huggingface.co/datasets/openai_humaneval) is to measure functional correctness for synthesizing programs from docstrings.
- [LongBench](https://github.com/THUDM/LongBench) is the first benchmark for bilingual, multitask, and comprehensive assessment of long context understanding capabilities of large language models.

## 7B Model Results

| Model             | **C-Eval** | **MMLU** | **CMMLU**  | **Gaokao**   | **AGIEval**   | **BBH**      | **GSM8K**   | **MATH**   | **HumanEval**   |
|:------------------|:-----------|:---------|:-----------|:-------------|:--------------|:-------------|:------------|:-----------|:----------------|
|                   | 5-shot     | 5-shot   | 5-shot     | 0-shot       | 0-shot        | 3-shot       | 4-shot      | 5-shot     | 0-shot          |
| GPT-4             | 69.9       | **86.4** | 71.2       | **72.3**     | **55.1**      | **86.7**     | **91.4**    | **45.8**   | **74.4**        |
| ChatGPT           | 52.5       | 70.0     | 53.9       | 51.1         | 39.9          | 70.1         | 78.2        | 28         | 73.2            |
| LLaMA2-7B         | 32.5       | 45.3     | 31.8       | 18.9         | 21.8          | 38.2         | 16.7        | 3.3        | 12.8            |
| ChatGLM2-6B(Base) | 51.7       | 47.9     | 50.0       | -            | -             | 33.7         | 32.4        | 6.5        | -               |
| Baichuan2-7B      | 56.3       | 54.7     | 57.0       | 34.8         | 34.6          | 41.8         | 24.6        | 5.4        | 17.7            |
| BlueLM-7B-Base    | 67.5       | **55.2** | 66.6       | **58.9**     | **43.4**      | 41.7         | 27.2        | 6.2        | 18.3            |
| BlueLM-7B-Chat    | **72.7**   | 50.7     | **74.2**   | 48.7         | **43.4**      | **65.6**     | **51.9**    | **13.4**   | **21.3**        |

## 7B-32K Model Results

We also tested our BlueLM-7B-Chat-32K  on the LongBench dataset and the results are shown in the table below:

| Model                  | Average  | Summary  | Single-Doc QA | Multi-Doc QA  | Code  | Few-shot | Synthetic |
|:-----------------------|:---------|:---------|:--------------|:--------------|:------|:---------|:----------|
| BlueLM-7B-Chat-32K     | 41.2     | 18.8     | 35.6          | 36.2          | 54.2  | 56.9     | 45.5      |

# Inference and Deployment

## Dependency Installation

You need to download this repository to use BlueLM:

```
git clone https://github.com/vivo-ai-lab/BlueLM
cd BlueLM
```

Install dependencies with pip:

```
pip install -r requirements.txt
```

When using BlueLM-7B-Base-32K or BlueLM-7B-Chat-32K, please install flash_attn additionally:

```
pip install flash_attn==2.3.3
```

If the installation fails, it is recommended to install [pre-build wheel file](https://github.com/Dao-AILab/flash-attention/releases/) of flash_attn.

## Usage

### Base Model

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("vivo-ai/BlueLM-7B-Base", trust_remote_code=True, use_fast=False)
>>> model = AutoModelForCausalLM.from_pretrained("vivo-ai/BlueLM-7B-Base", device_map="cuda:0", trust_remote_code=True)
>>> model = model.eval()
>>> inputs = tokenizer("儒林外史->吴敬梓\n隋唐演义->褚人获\n红楼梦->", return_tensors="pt")
>>> inputs = inputs.to("cuda:0")
>>> pred = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
>>> print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
儒林外史->吴敬梓
隋唐演义->褚人获
红楼梦->曹雪芹
三国演义->罗贯中
水浒传->施耐庵
西游记->吴承恩
聊斋志异->蒲松龄
封神演义->许仲琳
东周列国志->冯梦龙
三侠五义->石玉昆
七剑十三侠->唐芸洲
```

## Command Line Demo

![cli-demo](resources/cli_demo.gif)

```shell
python cli_demo.py
```

This command line demo is only designed for Chat Model.

## Web Demo

![web-demo](resources/web_demo.gif)

```shell
streamlit run web_demo.py --server.port 8080
```

This web demo is only designed for Chat Model.

## Demo Results of BlueLM-7B-Chat

<details><summary><b>Code</b></summary>

`````
User：某在线教育平台的课程数据存储在一个名为courses的列表中，每门课程用字典表示，包含课程名称（name）、教师姓名（teacher）和学生人数（students）三个字段。请编写一个Python函数，根据学生人数，计算课程的平均学生人数，并返回。

BlueLM-7B：下面是实现代码：

```python
def average_students(courses):
    total_students = 0
    count = 0
    for course in courses:
        total_students += course['students']
        count += 1
    if count == 0:
        return 0
    else:
        return total_students / count
```

这个函数接受一个课程列表作为参数，然后遍历列表中的每个课程，累加学生人数的总和，同时记录课程的数量。最后，如果列表为空，则返回0；否则，返回学生人数的平均值。
`````

</details>

<details><summary><b>Math</b></summary>

`````
User：朵朵有100元，她要买两本绘画书，绘画书每本40元，朵朵买完绘画书还剩多少元？

BlueLM-7B：可以先计算两本绘画书需要支付的总金额，然后计算购买后还剩下的金额。

已知朵朵有的钱数为：100元；

已知每本绘画书的价格为：40元；

因此，两本绘画书需要支付的总金额为：$40 \times 2 = 80$元；

购买两本绘画书后，还剩下的金额为：$100 - 80 = 20$元；

所以，朵朵买完绘画书还剩下20元。
`````

</details>

<details><summary><b>Role Play</b></summary>

`````
User：我是一名营销经理，最近我们推出了一款新产品，但是销售情况并不理想。这款产品是一款健康食品，主打低脂低糖，适合健身人群和需要控制饮食的人群。我们已经尝试过线上营销、线下推广、与知名健身房合作等方式，但是效果都不明显。你能帮我分析一下可能的原因以及提供一些销售策略吗？

BlueLM-7B：分析：

1. 竞争激烈：市场上已经存在许多类似的产品，消费者有更多的选择，导致新产品的市场占有率为零或很低。

2. 价格过高：该产品的价格比同类其他产品更高，可能会使消费者更倾向于购买其他品牌或更便宜的选项。

3. 缺乏宣传：没有足够的宣传使得消费者对该产品的了解程度不够，从而无法吸引更多的消费者。

4. 销售渠道不当：虽然已经尝试了多种销售方式，但可能没有选择到最适合的渠道，导致销售情况不佳。

策略：

1. 降低价格：考虑降低价格以吸引更多的消费者，同时保持产品质量和品质。

2. 加强宣传：增加宣传力度，包括在社交媒体上发布广告、增加线上营销等方式，提高消费者对产品的认知度。

3. 寻找合适的销售渠道：考虑与健身房等机构进行合作，或者寻找更符合目标消费者的销售渠道。

4. 改进产品：根据消费者的反馈意见，对产品进行改进，提高其质量和受欢迎程度。
`````
</details>

## Quantization Deployment

We release a pre-quantized 4bits version Chat Model: [BlueLM-7B-Chat-4bits](https://huggingface.co/vivo-ai/BlueLM-7B-Chat-4bits) for download. If your GPU memory is limited, you can try to load the model parameters with quantization. It only requires 5GB GPU memory. Moreover, BlueLM has been tested to smoothly generate text even under 4-bit quantization.

Install dependencies:

```
cd BlueLM/quant_cuda
python setup_cuda.py install
```

You can load and inference in the following manner:

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> tokenizer = AutoTokenizer.from_pretrained("vivo-ai/BlueLM-7B-Chat-4bits", trust_remote_code=True, use_fast=False)
>>> model = AutoModelForCausalLM.from_pretrained("vivo-ai/BlueLM-7B-Chat-4bits", device_map="cuda:0", trust_remote_code=True)
>>> model = model.eval()
>>> inputs = tokenizer("[|Human|]:三国演义的作者是谁？[|AI|]:", return_tensors="pt")
>>> inputs = inputs.to("cuda:0")
>>> outputs = model.generate(**inputs, max_new_tokens=128)
>>> print(tokenizer.decode(outputs.cpu()[0], skip_special_tokens=True))
三国演义的作者是谁？ 《三国演义》是由元末明初小说家罗贯中所著，是中国古典四大名著之一，也是中国古代历史小说发展的巅峰之作。
```

## Inference Acceleration

### Install vLLM

We added BlueLM inference code based on the [vllm](https://github.com/vllm-project/vllm) inference framework. The code is in the `example/vllm` directory.

The required version of the NVIDIA driver to install is 525.125.06, and the CUDA version should be 12.1.

```
python -m venv vllm
source vllm/bin/activate

cd example/vllm
pip install -e .
``` 

### Inference Demo

```
python vllm_demo.py
```

# Fine-tuning the Model

## Dependency Installation

```
pip install deepspeed==0.10.3
```

## Training Data

To demonstrate the fine-tuning process of our model in a simplified way, we selectively extracted 10,000 Chinese instruction data from the [BELLE project 500k Chinese instruction dataset](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) to serve as a demonstrative dataset. The data has been processed and can be accessed at `data/bella_train_demo.json` and `data/bella_dev_demo.json`.

## Fine-Tuning

After obtaining the processed data, you can perform fine-tuning training by configuring the corresponding paths and hyperparameters through the training script `script/bluelm-7b-sft.sh`.

The description of the relevant parameters is as follows:

| Parameter                       | Description                                                                                |
|:--------------------------------|:-------------------------------------------------------------------------------------------|
| **num_gpus**                    | Number of GPUs to use                                                                      |
| **train_file**                  | The path of train file                                                                     |
| **prompt_column**               | The name of prompt column                                                                  |
| **response_column**             | The name of response column                                                                |
| **model_name_or_path**          | Storage path for the preloaded model                                                       |
| **output_dir**                  | Storage path for the fine-tuned model                                                      |
| **tensorboard_dir**             | Storage path for the tensorboard                                                           |
| **seq_len**                     | Maximum length of training sequence                                                        |
| **batch_size_per_device**       | Number of samples per GPU input during training iteration                                  |
| **gradient_accumulation_steps** | Step length for gradient accumulation. Default is 1, which means no gradient accumulation  |
| **gradient_checkpointing**      | Whether to use gradient checkpointing                                                      |
| **max_steps**                   | Number of iterations for model training                                                    |
| **save_steps**                  | The interval step of model saving                                                          |
| **learning_rate**               | Learning rate                                                                              |
| **finetune**                    | Whether to enable fine-tuning                                                              |

You can start fine-tuning training using the following command:

```sh
cd train
sh script/bluelm-7b-sft.sh
```

## Fine-Tuning with LoRA

This project supports fine-tuning with LoRA. For detailed information about LoRA, please refer to the paper [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) and the Github repository [LoRA](https://github.com/microsoft/LoRA)。

The description of the relevant parameters is as follows:

| Parameter                       | Description                                                                              |
|:--------------------------------|:-----------------------------------------------------------------------------------------|
| **lora_rank**                   | The rank of the LoRA matrix. Generally set to 8, 16, 32, 64, etc                         |
| **lora_alpha**                  | LoRA alpha. Generally set to 16, 32, and so on.                                          |
| **lora_dropout**                | The dropout rate of the LoRA.                                                            |

You can start fine-tuning training with LoRA using the following command:

```sh
cd train
sh script/bluelm-7b-sft-lora.sh
```

# Disclaimer, License and Citation

## Disclaimer

We hereby declare strongly that all parties using the BlueLM models should not engage in any behavior that may damage national or social security, or violate relevant laws. We also request users not to use BlueLM model in product applications that have not been properly security-approved and registered. Please be sure to conduct all business activities under the premise of legality and compliance. We expect all users to comply with this.

This model is provided "as is". We have done our best to ensure the compliance of our traning data. Due to the complexity of the model and data, there may still be unforeseeable issues. We also strongly recommend users to conduct a detailed risk assessment of the model to ensure the legal compliance of the application. We will not assume any responsibility for any problems caused by using the BlueLM models.

## License

Our code is licensed under the [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) and [Community License for BlueLM Model](MODEL_LICENSE_EN.pdf). The BlueLM weights are completely open for academic research, and free commercial use is allowed after completing the [questionnaire](https://developers.vivo.com/buc/bluelm/apply).

## Citation

```
@misc{2023bluelm,
    title={BlueLM: An Open Multilingual 7B Language Model},
    author={BlueLM Team},
    howpublished = {\url{https://github.com/vivo-ai-lab/BlueLM}},
    year={2023}
}
```

# Contact Us

If you have any questions about the BlueLM, you can contact us via email (developers-ai@vivo.com). You can also scan the QR code to join the WeChat group.

![wechat](resources/wechat.png)
