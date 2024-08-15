<br>

### Notes

Of interest

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
