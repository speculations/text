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

<br>
<br>

<br>
<br>

<br>
<br>

