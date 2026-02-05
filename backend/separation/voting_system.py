"""
Ensemble Voting System
Select best stems from multiple model outputs using quality metrics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from .quality_metrics import QualityMetrics

logger = logging.getLogger(__name__)


class VotingSystem:
    """
    Intelligent stem selection using harmonic mean voting.

    For each stem, computes quality metrics across all model outputs
    and selects the best version based on harmonic mean of:
    - SNR (signal-to-noise ratio)
    - Cross-correlation (bleed detection)
    - Spectral entropy (complexity/naturalness)
    """

    def __init__(self,
                 sample_rate: int = 44100,
                 bleed_penalty_threshold: float = 0.4):
        """
        Args:
            sample_rate: Audio sample rate for analysis
            bleed_penalty_threshold: Cross-correlation above this gets penalized
        """
        self.metrics = QualityMetrics(sample_rate)
        self.bleed_penalty_threshold = bleed_penalty_threshold

    def vote_stems(self,
                  model_results: Dict[str, Dict[str, np.ndarray]],
                  reference_mix: np.ndarray = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Vote on best stem from each model's output.

        Args:
            model_results: Dict of {model_name: {stem_name: audio_array}}
            reference_mix: Original mixed audio (optional, for additional metrics)

        Returns:
            Tuple of (selected_stems, selection_report)
        """
        model_names = list(model_results.keys())
        if not model_names:
            raise ValueError("No model results to vote on")

        # Get stem names from first model
        stem_names = list(model_results[model_names[0]].keys())

        selected_stems = {}
        selection_report = {'models_compared': model_names, 'selections': {}}

        for stem_name in stem_names:
            logger.info(f"Voting on {stem_name}...")

            # Collect candidates from each model
            candidates = []
            for model_name in model_names:
                if stem_name in model_results[model_name]:
                    candidates.append({
                        'model': model_name,
                        'audio': model_results[model_name][stem_name]
                    })

            if not candidates:
                logger.warning(f"No candidates for {stem_name}")
                continue

            # Vote on best candidate
            best_idx, scores = self._select_best_stem(candidates, stem_name)

            selected_stems[stem_name] = candidates[best_idx]['audio']

            selection_report['selections'][stem_name] = {
                'selected_model': candidates[best_idx]['model'],
                'scores': scores
            }

            logger.info(f"  Selected {candidates[best_idx]['model']} for {stem_name} "
                       f"(score: {scores[best_idx]['harmonic_mean']:.3f})")

        return selected_stems, selection_report

    def _select_best_stem(self,
                         candidates: List[Dict],
                         stem_name: str) -> Tuple[int, List[Dict]]:
        """
        Select best stem from candidates using quality metrics.

        Returns:
            Tuple of (best_index, list_of_scores)
        """
        scores = []

        # Get all other stems for cross-correlation calculation
        # Use the first candidate's siblings as reference
        for candidate in candidates:
            audio = candidate['audio']

            # Create "other stems" from this model's output
            # (In full ensemble, these would be the other stems from same model)
            other_stems = {}
            for other_candidate in candidates:
                if other_candidate['model'] != candidate['model']:
                    # Use other models' versions as proxy for "other stems"
                    # This is a simplification; ideally we'd have all stems from each model
                    other_stems[other_candidate['model']] = other_candidate['audio']

            # If no other stems to compare, use self-analysis
            if not other_stems:
                # Create fake "other" by inverting phase
                other_stems['inverted'] = -audio * 0.5

            # Compute metrics
            metrics = self.metrics.compute_all_metrics(audio, other_stems)

            # Add penalty for high cross-correlation (bleed)
            if metrics['cross_correlation'] > self.bleed_penalty_threshold:
                penalty = (metrics['cross_correlation'] - self.bleed_penalty_threshold) * 0.5
                metrics['harmonic_mean'] *= (1.0 - penalty)
                metrics['bleed_penalty_applied'] = True
            else:
                metrics['bleed_penalty_applied'] = False

            metrics['model'] = candidate['model']
            scores.append(metrics)

        # Select best based on harmonic mean
        best_idx = np.argmax([s['harmonic_mean'] for s in scores])

        return best_idx, scores

    def compare_models(self,
                      model_results: Dict[str, Dict[str, np.ndarray]]) -> Dict:
        """
        Generate detailed comparison report across all models.

        Useful for debugging and A/B testing.
        """
        report = {
            'models': list(model_results.keys()),
            'per_stem_comparison': {},
            'overall_rankings': {}
        }

        model_names = list(model_results.keys())
        stem_names = list(model_results[model_names[0]].keys())

        model_total_scores = {m: 0.0 for m in model_names}

        for stem_name in stem_names:
            candidates = [
                {'model': m, 'audio': model_results[m][stem_name]}
                for m in model_names
                if stem_name in model_results[m]
            ]

            _, scores = self._select_best_stem(candidates, stem_name)

            report['per_stem_comparison'][stem_name] = {
                s['model']: {
                    'snr_db': s['snr_db'],
                    'cross_correlation': s['cross_correlation'],
                    'spectral_entropy': s['spectral_entropy'],
                    'harmonic_mean': s['harmonic_mean']
                }
                for s in scores
            }

            # Accumulate scores for overall ranking
            for s in scores:
                model_total_scores[s['model']] += s['harmonic_mean']

        # Overall rankings
        sorted_models = sorted(
            model_total_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        report['overall_rankings'] = {
            model: {'rank': i + 1, 'total_score': score}
            for i, (model, score) in enumerate(sorted_models)
        }

        return report
