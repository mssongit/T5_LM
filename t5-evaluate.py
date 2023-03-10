import sacrebleu
import logging
from rouge_score import rouge_scorer
from rouge_score import scoring


def rouge(targets,
          predictions,
          score_keys=("rouge1", "rouge2", "rougeLsum"),
          **kwargs):
  """Computes rouge score.
  Args:
    targets: list of strings
    predictions: list of strings
    score_keys: list of strings with the keys to compute.
    **kwargs: additional keyword arguments for RougeScorer.
  Returns:
    dict with score_key: rouge score across all targets and predictions
  """

  scorer = rouge_scorer.RougeScorer(rouge_types=score_keys, **kwargs)
  aggregator = scoring.BootstrapAggregator()

  def _prepare_summary(summary):
    # Make sure the summary is not bytes-type
    # Add newlines between sentences so that rougeLsum is computed correctly.
    summary = summary.replace(" . ", " .\n")
    return summary

  for prediction, target in zip(predictions, targets):
    target = _prepare_summary(target)
    prediction = _prepare_summary(prediction)
    aggregator.add_scores(scorer.score(target=target, prediction=prediction))
  result = aggregator.aggregate()
          
  for key in score_keys:
    logging.info(
        "%s = %.2f, 95%% confidence [%.2f, %.2f]",
        key,
        result[key].mid.fmeasure*100,
        result[key].low.fmeasure*100,
        result[key].high.fmeasure*100,
    )
  return {key: result[key].mid.fmeasure*100 for key in score_keys}
