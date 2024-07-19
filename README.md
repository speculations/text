<br>

## FOCUS

Use this baseline repository to investigate the items that can impact the mathematical performance of an abstractive text summarisation model, and to explore plausible business metrics.

<br>

Determine the direct or indirect effects of

* parameters
* hyperparameters

on automatic summarisation metrics; both data structuring and architecture related parameters and hyperparameters.  Of interest

* Recall-Oriented Understudy for Gisting Evaluation (ROUGE)
  * [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)
  * [Automatic Evaluation of Summaries Using N-gram Co-Occurrence Statistics](https://aclanthology.org/N03-1020.pdf)
  * [Automatic Evaluation of Machine Translation Quality Using Longest Common Subsequence and Skip-Bigram Statistics](https://aclanthology.org/P04-1077.pdf)
  * [Metric Card for ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge)
* [Measuring Bias in Contextualized Word Representations](https://arxiv.org/pdf/1906.07337)
* Aspects of [How to Compute the Probability of a Word](https://arxiv.org/pdf/2406.14561)
* Aspects of [Understanding Evaluation Metrics for Language Models](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/)

<br>

Determine the strengths and weaknesses of each metric, considering the problem in question, i.e., abstractive text summarisation; noting that the characteristics of a language, and its dialects, may impact the appropriateness of a metric.


<br>
<br>


## Upcoming

The production environments products, which lead on from the design herein, will address


### Products

* Saving tokenized data sets.
* Saving models.
* Default assets directories
  * Temporary Amazon Compute Machines
  * Amazon S3 (Simple Storage Service)
* Default container registry details for
  * Amazon Elastic Container Registry (ECR)
  * GitHub Container Registry (GCR)

<br>

### Warning Messages

These warning messages will be addressed:

* There were missing keys in the checkpoint model loaded: ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight'].
*  UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.

<br>

### Amazon Development & Deployment 

Launch templates, and orchestration templates, for model/system

* development
* deployment
* monitoring
* re-training

<br>
<br>


## References

[Example](https://huggingface.co/docs/transformers/tasks/summarization); [T5](https://huggingface.co/google-t5).

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


<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>
