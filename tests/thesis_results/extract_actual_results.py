"""
Extract Actual Results from EVEREST Experiments

This script extracts real experimental data from:
1. Production training results
2. Ablation study results
3. HPO study results
4. Model predictions and attention weights
5. Validation/test set statistics

Use this to replace simulated data with actual experimental results.
"""

from models.ablation.analysis import AblationAnalyzer
from models.training.analysis import ProductionAnalyzer
from models.solarknowledge_ret_plus import RETPlusWrapper
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings("ignore")

# Optional torch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class ActualResultsExtractor:
    """Extract actual experimental results for thesis."""

    def __init__(self):
        """Initialize the extractor."""
        self.output_dir = Path("actual_results")
        self.output_dir.mkdir(exist_ok=True)

        print("📊 EVEREST Actual Results Extractor")
        print("=" * 50)

    def extract_all_results(self):
        """Extract all actual experimental results."""
        print("\n🔍 Extracting actual experimental results...")

        # 1. Extract production training results
        production_results = self._extract_production_results()

        # 2. Extract ablation study results
        ablation_results = self._extract_ablation_results()

        # 3. Extract HPO results
        hpo_results = self._extract_hpo_results()

        # 4. Extract model predictions and attention
        prediction_results = self._extract_prediction_results()

        # 5. Extract dataset statistics
        dataset_stats = self._extract_dataset_statistics()

        # 6. Generate comprehensive summary
        self._generate_comprehensive_summary(
            production_results,
            ablation_results,
            hpo_results,
            prediction_results,
            dataset_stats,
        )

        print(f"\n✅ All actual results extracted to {self.output_dir}")

    def _extract_production_results(self) -> Dict[str, Any]:
        """Extract production training results."""
        print("\n📂 Extracting production training results...")

        results_dir = Path("models/training/results")
        if not results_dir.exists():
            print("⚠️ No production results directory found")
            return {}

        # Load all production experiments
        production_data = []

        for result_file in results_dir.glob("*.json"):
            try:
                with open(result_file, "r") as f:
                    result = json.load(f)

                # Extract key metrics
                experiment_data = {
                    "experiment_name": result.get("experiment_name", result_file.stem),
                    "flare_class": self._extract_flare_class(result_file.stem),
                    "time_window": self._extract_time_window(result_file.stem),
                    "seed": self._extract_seed(result_file.stem),
                    "final_epoch": result.get("final_epoch", 0),
                    "training_time": result.get("training_time", 0),
                    "best_val_tss": result.get("best_val_tss", 0),
                    "test_metrics": result.get("test_metrics", {}),
                    "threshold_analysis": result.get("threshold_analysis", {}),
                    "model_path": result.get("model_path", ""),
                    "config": result.get("config", {}),
                }

                # Extract test metrics
                test_metrics = result.get("test_metrics", {})
                for metric in [
                    "tss",
                    "f1",
                    "precision",
                    "recall",
                    "roc_auc",
                    "brier",
                    "ece",
                ]:
                    experiment_data[metric] = test_metrics.get(metric, 0)

                # Extract optimal threshold
                threshold_analysis = result.get("threshold_analysis", {})
                experiment_data["optimal_threshold"] = threshold_analysis.get(
                    "optimal_threshold", 0.5
                )
                experiment_data["latency_ms"] = result.get("inference_latency_ms", 0)

                production_data.append(experiment_data)

            except Exception as e:
                print(f"⚠️ Error loading {result_file}: {e}")
                continue

        # Convert to DataFrame
        df_production = pd.DataFrame(production_data)

        # Save extracted data
        df_production.to_csv(self.output_dir / "production_results.csv", index=False)

        print(f"✅ Extracted {len(df_production)} production experiments")
        return {"dataframe": df_production, "raw_data": production_data}

    def _extract_ablation_results(self) -> Dict[str, Any]:
        """Extract ablation study results."""
        print("\n📂 Extracting ablation study results...")

        results_dir = Path("models/ablation/results")
        if not results_dir.exists():
            print("⚠️ No ablation results directory found")
            return {}

        # Load ablation analyzer
        try:
            analyzer = AblationAnalyzer()
            analyzer.load_all_results()
            analyzer.aggregate_results()
            analyzer.perform_statistical_tests()

            # Extract key results
            ablation_data = {
                "raw_results": analyzer.results,
                "aggregated_results": analyzer.aggregated_results,
                "statistical_tests": analyzer.statistical_tests,
                "summary_stats": self._compute_ablation_summary(analyzer),
            }

            # Save extracted data
            with open(self.output_dir / "ablation_results.json", "w") as f:
                json.dump(ablation_data, f, indent=2, default=str)

            print(f"✅ Extracted ablation results for {len(analyzer.results)} variants")
            return ablation_data

        except Exception as e:
            print(f"⚠️ Error extracting ablation results: {e}")
            return {}

    def _extract_hpo_results(self) -> Dict[str, Any]:
        """Extract HPO study results."""
        print("\n📂 Extracting HPO study results...")

        hpo_dir = Path("models/hpo/results")
        if not hpo_dir.exists():
            print("⚠️ No HPO results directory found")
            return {}

        hpo_data = {}

        # Load HPO study results
        for result_file in hpo_dir.glob("*.json"):
            try:
                with open(result_file, "r") as f:
                    result = json.load(f)

                target = result_file.stem
                hpo_data[target] = {
                    "best_params": result.get("best_params", {}),
                    "best_score": result.get("best_score", 0),
                    "n_trials": result.get("n_trials", 0),
                    "study_duration": result.get("study_duration", 0),
                    "convergence_trial": result.get("convergence_trial", 0),
                    "param_importance": result.get("param_importance", {}),
                    "optimization_history": result.get("optimization_history", []),
                }

            except Exception as e:
                print(f"⚠️ Error loading HPO result {result_file}: {e}")
                continue

        # Save extracted data
        with open(self.output_dir / "hpo_results.json", "w") as f:
            json.dump(hpo_data, f, indent=2, default=str)

        print(f"✅ Extracted HPO results for {len(hpo_data)} targets")
        return hpo_data

    def _extract_prediction_results(self) -> Dict[str, Any]:
        """Extract model predictions and attention weights."""
        print("\n📂 Extracting model predictions and attention...")

        prediction_data = {}

        # Look for saved predictions
        predictions_dir = Path("models/predictions")
        if predictions_dir.exists():
            for pred_file in predictions_dir.glob("*.pkl"):
                try:
                    with open(pred_file, "rb") as f:
                        predictions = pickle.load(f)

                    target = pred_file.stem
                    prediction_data[target] = {
                        "predictions": predictions.get("predictions", []),
                        "true_labels": predictions.get("true_labels", []),
                        "probabilities": predictions.get("probabilities", []),
                        "attention_weights": predictions.get("attention_weights", []),
                        "sample_ids": predictions.get("sample_ids", []),
                        "metadata": predictions.get("metadata", {}),
                    }

                except Exception as e:
                    print(f"⚠️ Error loading predictions {pred_file}: {e}")
                    continue

        # Extract attention patterns for key samples
        attention_analysis = self._analyze_attention_patterns(prediction_data)

        # Save extracted data
        with open(self.output_dir / "prediction_results.json", "w") as f:
            json.dump(prediction_data, f, indent=2, default=str)

        with open(self.output_dir / "attention_analysis.json", "w") as f:
            json.dump(attention_analysis, f, indent=2, default=str)

        print(f"✅ Extracted predictions for {len(prediction_data)} targets")
        return {
            "predictions": prediction_data,
            "attention_analysis": attention_analysis,
        }

    def _extract_dataset_statistics(self) -> Dict[str, Any]:
        """Extract dataset statistics for run matrix table."""
        print("\n📂 Extracting dataset statistics...")

        # Look for dataset statistics
        data_dir = Path("data")
        stats_file = data_dir / "dataset_statistics.json"

        if stats_file.exists():
            with open(stats_file, "r") as f:
                dataset_stats = json.load(f)
        else:
            # Generate from available data files
            dataset_stats = self._compute_dataset_statistics()

        # Save extracted data
        with open(self.output_dir / "dataset_statistics.json", "w") as f:
            json.dump(dataset_stats, f, indent=2)

        print(f"✅ Extracted dataset statistics")
        return dataset_stats

    def _compute_dataset_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics from available data."""
        # This would analyze your actual data files
        # For now, return structure with placeholders

        flare_classes = ["C", "M", "M5"]
        time_windows = [24, 48, 72]

        stats = {}

        for flare_class in flare_classes:
            for time_window in time_windows:
                key = f"{flare_class}_{time_window}h"

                # These would be computed from actual data
                stats[key] = {
                    "train_positive": 0,
                    "train_negative": 0,
                    "val_positive": 0,
                    "val_negative": 0,
                    "test_positive": 0,
                    "test_negative": 0,
                    "class_ratio": 0.0,
                    "total_samples": 0,
                }

        return stats

    def _analyze_attention_patterns(
        self, prediction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze attention patterns for key samples."""
        attention_analysis = {}

        for target, data in prediction_data.items():
            if "attention_weights" not in data or not data["attention_weights"]:
                continue

            attention_weights = np.array(data["attention_weights"])
            predictions = np.array(data["predictions"])
            true_labels = np.array(data["true_labels"])

            # Find representative samples
            tp_indices = np.where((predictions == 1) & (true_labels == 1))[0]
            tn_indices = np.where((predictions == 0) & (true_labels == 0))[0]
            fp_indices = np.where((predictions == 1) & (true_labels == 0))[0]
            fn_indices = np.where((predictions == 0) & (true_labels == 1))[0]

            analysis = {}

            for sample_type, indices in [
                ("true_positive", tp_indices),
                ("true_negative", tn_indices),
                ("false_positive", fp_indices),
                ("false_negative", fn_indices),
            ]:
                if len(indices) > 0:
                    # Take first few samples
                    sample_indices = indices[:5]
                    sample_attention = attention_weights[sample_indices]

                    analysis[sample_type] = {
                        "sample_indices": sample_indices.tolist(),
                        "attention_patterns": sample_attention.tolist(),
                        "mean_attention": np.mean(sample_attention, axis=0).tolist(),
                        "std_attention": np.std(sample_attention, axis=0).tolist(),
                    }

            attention_analysis[target] = analysis

        return attention_analysis

    def _compute_ablation_summary(self, analyzer) -> Dict[str, Any]:
        """Compute summary statistics for ablation study."""
        summary = {
            "total_variants": len(analyzer.results),
            "total_experiments": sum(
                len(variant_results) for variant_results in analyzer.results.values()
            ),
            "significant_effects": 0,
            "largest_effect": {"variant": "", "effect_size": 0, "metric": ""},
            "most_important_components": [],
        }

        # Count significant effects
        if hasattr(analyzer, "statistical_tests"):
            for variant, tests in analyzer.statistical_tests.items():
                for metric, test_result in tests.items():
                    if test_result.get("is_significant", False):
                        summary["significant_effects"] += 1

                        # Track largest effect
                        effect_size = abs(test_result.get("observed_diff", 0))
                        if effect_size > summary["largest_effect"]["effect_size"]:
                            summary["largest_effect"] = {
                                "variant": variant,
                                "effect_size": effect_size,
                                "metric": metric,
                            }

        return summary

    def _extract_flare_class(self, filename: str) -> str:
        """Extract flare class from filename."""
        if "M5" in filename:
            return "M5"
        elif "M_" in filename or "_M_" in filename:
            return "M"
        elif "C_" in filename or "_C_" in filename:
            return "C"
        return "unknown"

    def _extract_time_window(self, filename: str) -> int:
        """Extract time window from filename."""
        if "72h" in filename:
            return 72
        elif "48h" in filename:
            return 48
        elif "24h" in filename:
            return 24
        return 0

    def _extract_seed(self, filename: str) -> int:
        """Extract seed from filename."""
        import re

        match = re.search(r"seed(\d+)", filename)
        return int(match.group(1)) if match else 0

    def _generate_comprehensive_summary(
        self,
        production_results,
        ablation_results,
        hpo_results,
        prediction_results,
        dataset_stats,
    ):
        """Generate comprehensive summary of all results."""
        print("\n📋 Generating comprehensive summary...")

        summary = {
            "extraction_timestamp": pd.Timestamp.now().isoformat(),
            "production_training": {
                "total_experiments": len(production_results.get("dataframe", [])),
                "completed_targets": len(
                    production_results.get("dataframe", pd.DataFrame()).groupby(
                        ["flare_class", "time_window"]
                    )
                )
                if "dataframe" in production_results
                else 0,
                "best_performance": self._get_best_performance(production_results),
                "average_training_time": self._get_average_training_time(
                    production_results
                ),
            },
            "ablation_study": {
                "total_variants": len(ablation_results.get("raw_results", {})),
                "significant_effects": ablation_results.get("summary_stats", {}).get(
                    "significant_effects", 0
                ),
                "largest_effect": ablation_results.get("summary_stats", {}).get(
                    "largest_effect", {}
                ),
            },
            "hpo_study": {
                "total_targets": len(hpo_results),
                "average_trials": np.mean(
                    [
                        target_data.get("n_trials", 0)
                        for target_data in hpo_results.values()
                    ]
                )
                if hpo_results
                else 0,
                "best_hyperparameters": self._get_best_hyperparameters(hpo_results),
            },
            "predictions": {
                "total_targets": len(prediction_results.get("predictions", {})),
                "attention_samples": sum(
                    len(target_data.get("attention_weights", []))
                    for target_data in prediction_results.get(
                        "predictions", {}
                    ).values()
                ),
            },
            "dataset": {
                "total_targets": len(dataset_stats),
                "class_imbalance_range": self._get_class_imbalance_range(dataset_stats),
            },
        }

        # Save comprehensive summary
        with open(self.output_dir / "comprehensive_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"✅ Comprehensive summary saved")
        return summary

    def _get_best_performance(self, production_results) -> Dict[str, Any]:
        """Get best performance across all experiments."""
        if (
            "dataframe" not in production_results
            or len(production_results["dataframe"]) == 0
        ):
            return {}

        df = production_results["dataframe"]
        best_tss_idx = df["tss"].idxmax()
        best_experiment = df.loc[best_tss_idx]

        return {
            "experiment": best_experiment["experiment_name"],
            "tss": best_experiment["tss"],
            "flare_class": best_experiment["flare_class"],
            "time_window": best_experiment["time_window"],
        }

    def _get_average_training_time(self, production_results) -> float:
        """Get average training time."""
        if (
            "dataframe" not in production_results
            or len(production_results["dataframe"]) == 0
        ):
            return 0.0

        df = production_results["dataframe"]
        return df["training_time"].mean() if "training_time" in df.columns else 0.0

    def _get_best_hyperparameters(self, hpo_results) -> Dict[str, Any]:
        """Get best hyperparameters across targets."""
        if not hpo_results:
            return {}

        # Find target with best score
        best_target = max(
            hpo_results.keys(), key=lambda k: hpo_results[k].get("best_score", 0)
        )

        return hpo_results[best_target].get("best_params", {})

    def _get_class_imbalance_range(self, dataset_stats) -> Dict[str, float]:
        """Get range of class imbalance ratios."""
        if not dataset_stats:
            return {}

        ratios = [stats.get("class_ratio", 0) for stats in dataset_stats.values()]
        ratios = [r for r in ratios if r > 0]

        if not ratios:
            return {}

        return {
            "min_ratio": min(ratios),
            "max_ratio": max(ratios),
            "mean_ratio": np.mean(ratios),
        }


def main():
    """Main function to extract all actual results."""
    extractor = ActualResultsExtractor()
    extractor.extract_all_results()

    print("\n🎉 Actual results extraction complete!")
    print("\nExtracted files:")
    print("📊 Data:")
    print("   - production_results.csv")
    print("   - ablation_results.json")
    print("   - hpo_results.json")
    print("   - prediction_results.json")
    print("   - attention_analysis.json")
    print("   - dataset_statistics.json")
    print("   - comprehensive_summary.json")
    print("\n💡 Next steps:")
    print("1. Review extracted data for completeness")
    print("2. Run generate_publication_results.py with actual data")
    print("3. Generate missing analysis components")
    print("4. Compile final thesis results")


if __name__ == "__main__":
    main()
