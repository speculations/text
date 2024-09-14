<br>

## Environments

### Remote Development

For this Python project/template, the remote development environment requires

* [Dockerfile](../.devcontainer/Dockerfile)
* [requirements.txt](../.devcontainer/requirements.txt)

An image is built via the command

```shell
docker build . --file .devcontainer/Dockerfile -t text
```

On success, the output of

```shell
docker images
```

should include

<br>

| repository | tag    | image id | created  | size     |
|:-----------|:-------|:---------|:---------|:---------|
| text       | latest | $\ldots$ | $\ldots$ | $\ldots$ |


<br>

Subsequently, run a container, i.e., an instance, of the image `text` via:

<br>

```shell
docker run --rm --gpus all --shm-size=16gb -i -t 
  -p 127.0.0.1:6007:6007 -p 127.0.0.1:6006:6006 
    -p 172.17.0.2:8265:8265 -p 172.17.0.2:6379:6379 -w /app 
	    --mount type=bind,src="$(pwd)",target=/app text
```

or

```shell
docker run --rm --gpus all --shm-size=16gb -i -t 
  -p 6007:6007 -p 6006:6006 -p 8265:8265 -p 6379:6379  
    -w /app --mount type=bind,src="$(pwd)",target=/app text
```

<br>

Herein, `-p 6007:6007` maps the host port `6007` to container port `6007`.  Note, the container's working environment, i.e., -w, must be inline with this project's top directory.  Additionally

* --rm: [automatically remove container](https://docs.docker.com/engine/reference/commandline/run/#:~:text=a%20container%20exits-,%2D%2Drm,-Automatically%20remove%20the)
* -i: [interact](https://docs.docker.com/engine/reference/commandline/run/#:~:text=and%20reaps%20processes-,%2D%2Dinteractive,-%2C%20%2Di)
* -t: [tag](https://docs.docker.com/get-started/02_our_app/#:~:text=Finally%2C%20the-,%2Dt,-flag%20tags%20your)
* -p: [publish](https://docs.docker.com/engine/reference/commandline/run/#:~:text=%2D%2Dpublish%20%2C-,%2Dp,-Publish%20a%20container%E2%80%99s)

<br>

Get the name of the running instance of ``text`` via:

```shell
docker ps --all
```

Never deploy a root container.

<br>


### Remote Development & Integrated Development Environments

An IDE (integrated development environment) is a helpful remote development tool.  The **IntelliJ
IDEA** set up involves connecting to a machine's Docker [daemon](https://www.jetbrains.com/help/idea/docker.html#connect_to_docker), the steps are

<br>

> * **Settings** $\rightarrow$ **Build, Execution, Deployment** $\rightarrow$ **Docker** $\rightarrow$ **WSL:** {select the linux operating system}
> * **View** $\rightarrow$ **Tool Window** $\rightarrow$ **Services** <br>Within the **Containers** section connect to the running instance of interest, or ascertain connection to the running instance of interest.

<br>

**Visual Studio Code** has its container attachment instructions; study [Attach Container](https://code.visualstudio.com/docs/devcontainers/attach-container).


<br>
<br>

## Code Analysis

The GitHub Actions script [main.yml](../.github/workflows/main.yml) conducts code analysis within a Cloud GitHub Workspace.  Depending on the script, code analysis may occur `on push` to any repository branch, or `on push` to a specific branch.

The sections herein outline remote code analysis.

### pylint

The directive

```shell
pylint --generate-rcfile > .pylintrc
```

generates the dotfile `.pylintrc` of the static code analyser [pylint](https://pylint.pycqa.org/en/latest/user_guide/checkers/features.html).  Analyse a directory via the command

```shell
python -m pylint --rcfile .pylintrc {directory}
```

The `.pylintrc` file of this template project has been **amended to adhere to team norms**, including

* Maximum number of characters on a single line.
  > max-line-length=127

* Maximum number of lines in a module.
  > max-module-lines=135


<br>

### pytest & pytest coverage

The directive patterns

```shell
python -m pytest tests/{directory.name}/...py
pytest --cov-report term-missing  --cov src/{directory.name}/...py tests/{directory.name}/...py
```

for test and test coverage, respectively.

<br>

### flake8

For code & complexity analysis.  A directive of the form

```bash
python -m flake8 --count --select=E9,F63,F7,F82 --show-source --statistics src/...
```

inspects issues in relation to logic (F7), syntax (Python E9, Flake F7), mathematical formulae symbols (F63), undefined variable names (F82).  Additionally

```shell
python -m flake8 --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics src/...
```

inspects complexity.

<br>
<br>

## Model Development

### Metrics

* Recall-Oriented Understudy for Gisting Evaluation (ROUGE)
  * [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)
  * [Automatic Evaluation of Summaries Using N-gram Co-Occurrence Statistics](https://aclanthology.org/N03-1020.pdf)
  * [Automatic Evaluation of Machine Translation Quality Using Longest Common Subsequence and Skip-Bigram Statistics](https://aclanthology.org/P04-1077.pdf)
  * [Metric Card for ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge)
* [Measuring Bias in Contextualized Word Representations](https://arxiv.org/pdf/1906.07337)
* Aspects of [How to Compute the Probability of a Word](https://arxiv.org/pdf/2406.14561)
* Aspects of [Understanding Evaluation Metrics for Language Models](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/)

<br>

### Warning Messages

These warning messages will be addressed:

* There were missing keys in the checkpoint model loaded: ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight'].
* UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.


<br>
<br>

## References

Articles:

* [T5: Text-To-Text Transfer Transformer](https://huggingface.co/docs/transformers/tasks/summarization); [T5](https://huggingface.co/google-t5).
* [Population Based Training](https://deepmind.google/discover/blog/population-based-training-of-neural-networks/), ([paper](https://arxiv.org/abs/1711.09846))

<br>

Critical Classes & Utilities:

* [Data Splitting](https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/main_classes#datasets.Dataset.train_test_split)
* [AutoModel.from_pretrained](https://huggingface.co/docs/transformers/v4.42.0/en/model_doc/auto#transformers.AutoModel.from_pretrained)
  * [pre-trained configuration](https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/configuration#transformers.PretrainedConfig)
  * [PreTrainedTokenizerFast](https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast)
* [Text Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
  * Beware, model generation configuration settings are undergoing changes.  Instead: [default text generation configuration.](https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration)
  * [generation configuration](https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/text_generation#transformers.GenerationConfig)
  * [from_pretrained](https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/text_generation#transformers.GenerationConfig.from_pretrained)
* Modelling
  * [Seq2SeqTrainer](https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/trainer#transformers.Seq2SeqTrainer)
  * [metrics & batch_decode](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.batch_decode)
  * [rouge](https://huggingface.co/spaces/evaluate-metric/rouge)
  * [Utilities for Trainer: EvalPrediction](https://huggingface.co/docs/transformers/v4.42.0/en/internal/trainer_utils#transformers.EvalPrediction)
  * [Evaluate](https://huggingface.co/docs/evaluate/index).  [An old approach; glue_metric_compute.](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/trainer/01_text_classification.ipynb)

<br>

Hyperparameters

* [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
* [Using Huggingface Transformers with Tune](https://docs.ray.io/en/latest/tune/examples/pbt_transformers.html)
  * [Configure PBT and Tuner](https://docs.ray.io/en/latest/tune/examples/pbt_visualization/pbt_visualization.html?_gl=1*13cvafe*_ga*MTY2MzU4MzU2OC4xNzIzNjU3NDI2*_up*MQ..*_ga_0LCWHW1N3S*MTcyMzY1NzQyNi4xLjAuMTcyMzY1NzQyNi4wLjAuMA..#configure-pbt-and-tuner)
  * [ray.tune.schedulers.PopulationBasedTraining](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html), [schedulers](https://docs.ray.io/en/latest/tune/api/schedulers.html)
  * [ray.tune.Tuner](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.html)
  * [tune_basic_example](https://docs.ray.io/en/latest/tune/examples/includes/tune_basic_example.html)
  * [A Guide To Parallelism and Resources for Ray Tune](https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html)
  * [Get Started with Distributed Training using Hugging Face Transformers](https://docs.ray.io/en/latest/train/getting-started-transformers.html)
* [Logging and Outputs in Tune](https://docs.ray.io/en/latest/tune/tutorials/tune-output.html)
  * And, using TensorBoard
* Pytorch
  * [TensorboardX](https://tensorboardx.readthedocs.io/en/latest/tutorial.html#what-is-tensorboard-x)

<br>

System

* [Ray Dashboard](https://docs.ray.io/en/latest/ray-observability/getting-started.html)
* [Ray, Grafana, Prometheus](https://docs.ray.io/en/latest/cluster/configure-manage-dashboard.html#embed-grafana-visualizations-into-ray-dashboard)


<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>
