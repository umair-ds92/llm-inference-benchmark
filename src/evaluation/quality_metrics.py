"""
Quality Metrics for LLM Optimization Evaluation

Measures quality degradation caused by quantization and other optimizations.

Metrics used:
- Task Accuracy:  % of correct answers on Q&A workload
- ROUGE Score:    Overlap between generated and reference summaries
- Perplexity:     Lower = better; measures how "surprised" the model is
- Consistency:    Same prompt → same answer across quantization levels
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re
from src.utils import get_logger

logger = get_logger(__name__)

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge-score not installed. Run: pip install rouge-score")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class QualityResult:
    """Result of quality evaluation for one model/config"""
    model_name: str
    quantization: str
    task_accuracy: Optional[float] = None     # 0.0 – 1.0
    rouge_l_score: Optional[float] = None     # 0.0 – 1.0
    consistency_score: Optional[float] = None # 0.0 – 1.0
    num_samples: int = 0

    # vs FP16 baseline
    accuracy_delta: Optional[float] = None    # negative = degradation
    rouge_delta: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name":        self.model_name,
            "quantization":      self.quantization,
            "task_accuracy":     self.task_accuracy,
            "rouge_l_score":     self.rouge_l_score,
            "consistency_score": self.consistency_score,
            "num_samples":       self.num_samples,
            "accuracy_delta":    self.accuracy_delta,
            "rouge_delta":       self.rouge_delta,
        }

    def quality_degradation_pct(self) -> Optional[float]:
        """Overall quality degradation as a percentage (positive = worse)"""
        if self.accuracy_delta is not None:
            return round(-self.accuracy_delta * 100, 2)
        return None


class QualityEvaluator:
    """
    Evaluate quality of LLM outputs.

    Usage:
        evaluator = QualityEvaluator()

        # Evaluate task accuracy (Q&A)
        result = evaluator.evaluate_accuracy(
            model, prompts, ground_truth_answers
        )

        # Evaluate summarization quality (ROUGE)
        result = evaluator.evaluate_rouge(
            model, article_prompts, reference_summaries
        )
    """

    # Simple keyword-based answer checker for Q&A evaluation
    QA_ANSWERS: Dict[str, List[str]] = {
        "What is the capital of France?":                  ["paris"],
        "Who wrote Romeo and Juliet?":                     ["shakespeare"],
        "What is the speed of light?":                     ["299", "300", "3×10"],
        "When was the Declaration of Independence signed?": ["1776"],
        "What is the largest planet in our solar system?": ["jupiter"],
        "Who painted the Mona Lisa?":                      ["da vinci", "leonardo"],
        "What is the chemical symbol for gold?":           ["au"],
        "How many continents are there?":                  ["7", "seven"],
        "What is the boiling point of water in Celsius?":  ["100"],
        "Who invented the telephone?":                     ["bell", "alexander"],
        "What is the currency of Japan?":                  ["yen"],
        "When did World War II end?":                      ["1945"],
        "What is the square root of 144?":                 ["12"],
        "Who was the first president of the United States?":["washington"],
        "What is photosynthesis?":                         ["light", "chlorophyll", "plants"],
        "How many bones are in the human body?":           ["206"],
        "What is the capital of Australia?":               ["canberra"],
        "Who wrote The Great Gatsby?":                     ["fitzgerald"],
        "What is the formula for water?":                  ["h2o", "h₂o"],
        "When was the internet invented?":                 ["1960", "1970", "arpanet"],
    }

    def evaluate_accuracy(
        self,
        model,
        prompts: List[str],
        max_tokens: int = 100,
    ) -> QualityResult:
        """
        Evaluate Q&A task accuracy.

        Args:
            model:      Loaded model instance
            prompts:    Q&A prompts (must be in QA_ANSWERS dict)
            max_tokens: Max tokens to generate per answer

        Returns:
            QualityResult with task_accuracy populated
        """
        correct = 0
        total   = 0

        for prompt in prompts:
            expected_keywords = self.QA_ANSWERS.get(prompt)
            if expected_keywords is None:
                logger.debug(f"No ground truth for: {prompt[:50]}...")
                continue

            try:
                response = model.generate(prompt, max_tokens=max_tokens)
                answer   = response.text.lower()

                is_correct = any(kw in answer for kw in expected_keywords)
                if is_correct:
                    correct += 1
                total += 1

            except Exception as e:
                logger.error(f"Error evaluating: {e}")
                continue

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Accuracy: {correct}/{total} = {accuracy:.2%}")

        return QualityResult(
            model_name=getattr(model, "name", "unknown"),
            quantization="unknown",
            task_accuracy=round(accuracy, 4),
            num_samples=total,
        )

    def evaluate_rouge(
        self,
        generated_texts: List[str],
        reference_texts: List[str],
        model_name: str = "unknown",
        quantization: str = "unknown",
    ) -> QualityResult:
        """
        Evaluate summarization quality using ROUGE-L.

        Args:
            generated_texts: Model-generated summaries
            reference_texts: Reference (ground-truth) summaries
            model_name:      Model identifier
            quantization:    Quantization level used

        Returns:
            QualityResult with rouge_l_score populated
        """
        if not ROUGE_AVAILABLE:
            logger.warning("rouge-score not available. Skipping ROUGE evaluation.")
            return QualityResult(
                model_name=model_name,
                quantization=quantization,
                num_samples=len(generated_texts),
            )

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = []

        for gen, ref in zip(generated_texts, reference_texts):
            result = scorer.score(ref, gen)
            scores.append(result["rougeL"].fmeasure)

        avg_rouge = sum(scores) / len(scores) if scores else 0.0
        logger.info(f"ROUGE-L: {avg_rouge:.4f} over {len(scores)} samples")

        return QualityResult(
            model_name=model_name,
            quantization=quantization,
            rouge_l_score=round(avg_rouge, 4),
            num_samples=len(scores),
        )

    def evaluate_consistency(
        self,
        model,
        prompts: List[str],
        num_runs: int = 3,
        temperature: float = 0.0,
    ) -> QualityResult:
        """
        Measure how consistent model outputs are across repeated runs.
        (At temperature=0 outputs should be identical; divergence
         signals instability from aggressive quantization.)

        Args:
            model:       Loaded model instance
            prompts:     Prompts to test
            num_runs:    How many times to repeat each prompt
            temperature: Sampling temperature (0 = deterministic)

        Returns:
            QualityResult with consistency_score populated
        """
        consistent_count = 0
        total_count      = 0

        for prompt in prompts[:20]:  # Cap at 20 prompts
            outputs = []
            for _ in range(num_runs):
                try:
                    resp = model.generate(prompt, temperature=temperature, max_tokens=100)
                    outputs.append(resp.text.strip())
                except Exception as e:
                    logger.error(f"Error: {e}")

            if len(outputs) >= 2:
                # Check if all outputs are the same
                is_consistent = len(set(outputs)) == 1
                if is_consistent:
                    consistent_count += 1
                total_count += 1

        score = consistent_count / total_count if total_count > 0 else 0.0
        logger.info(f"Consistency: {consistent_count}/{total_count} = {score:.2%}")

        return QualityResult(
            model_name=getattr(model, "name", "unknown"),
            quantization="unknown",
            consistency_score=round(score, 4),
            num_samples=total_count,
        )

    def compare_quality(
        self,
        baseline_result: QualityResult,
        optimized_result: QualityResult,
    ) -> QualityResult:
        """
        Compare optimized result against FP16 baseline.

        Populates *_delta fields (negative = degradation vs baseline).
        """
        optimized_result.accuracy_delta = (
            (optimized_result.task_accuracy or 0)
            - (baseline_result.task_accuracy or 0)
        )
        optimized_result.rouge_delta = (
            (optimized_result.rouge_l_score or 0)
            - (baseline_result.rouge_l_score or 0)
        )
        return optimized_result