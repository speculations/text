<br>

### Notes

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

