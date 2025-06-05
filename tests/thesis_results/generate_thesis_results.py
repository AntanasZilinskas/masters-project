from models.create_missing_analysis import MissingAnalysisGenerator
from models.generate_publication_results import PublicationResultsGenerator
from models.extract_actual_results import ActualResultsExtractor
import pandas as pd

#!/usr/bin/env python3
"""
Generate all thesis results, tables, and figures.

Usage:
    python generate_thesis_results.py [--mode MODE] [--output-dir DIR]

Modes:
    - full: Complete generation (default)
    - extract: Only extract actual results
    - publish: Only generate publication materials
    - missing: Only generate missing components
    - validate: Only validate completeness
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import warnings

warnings.filterwarnings("ignore")

# Add models directory to path
sys.path.append(str(Path(__file__).parent / "models"))


class ThesisResultsOrchestrator:
    """Master orchestrator for thesis results generation."""

    def __init__(self, output_dir: str = "thesis_results"):
        """Initialize the orchestrator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.actual_results_dir = self.output_dir / "actual_results"
        self.publication_dir = self.output_dir / "publication_results"
        self.missing_analysis_dir = self.output_dir / "missing_analysis"
        self.validation_dir = self.output_dir / "validation"

        for dir_path in [
            self.actual_results_dir,
            self.publication_dir,
            self.missing_analysis_dir,
            self.validation_dir,
        ]:
            dir_path.mkdir(exist_ok=True)

        print("🎓 EVEREST Thesis Results Generator")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Timestamp: {pd.Timestamp.now()}")

    def generate_all_results(self):
        """Generate all thesis results in the correct order."""
        print("\n🚀 Starting complete thesis results generation...")

        try:
            # Step 1: Extract actual experimental data
            print("\n" + "=" * 60)
            print("STEP 1: EXTRACTING ACTUAL EXPERIMENTAL DATA")
            print("=" * 60)
            actual_data = self._extract_actual_results()

            # Step 2: Generate publication materials
            print("\n" + "=" * 60)
            print("STEP 2: GENERATING PUBLICATION MATERIALS")
            print("=" * 60)
            publication_results = self._generate_publication_materials()

            # Step 3: Create missing analysis components
            print("\n" + "=" * 60)
            print("STEP 3: CREATING MISSING ANALYSIS COMPONENTS")
            print("=" * 60)
            missing_components = self._create_missing_components()

            # Step 4: Validate completeness
            print("\n" + "=" * 60)
            print("STEP 4: VALIDATING COMPLETENESS")
            print("=" * 60)
            validation_report = self._validate_completeness()

            # Step 5: Generate final summary
            print("\n" + "=" * 60)
            print("STEP 5: GENERATING FINAL SUMMARY")
            print("=" * 60)
            final_summary = self._generate_final_summary(
                actual_data, publication_results, missing_components, validation_report
            )

            print("\n🎉 THESIS RESULTS GENERATION COMPLETE!")
            self._print_final_report(final_summary)

        except Exception as e:
            print(f"\n❌ Error during results generation: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    def _extract_actual_results(self) -> Dict[str, Any]:
        """Extract actual experimental results."""
        print("📊 Extracting actual experimental data...")

        # Change to actual results directory
        original_cwd = os.getcwd()

        try:
            extractor = ActualResultsExtractor()
            extractor.output_dir = self.actual_results_dir
            extractor.extract_all_results()

            # Load summary
            summary_file = self.actual_results_dir / "comprehensive_summary.json"
            if summary_file.exists():
                with open(summary_file, "r") as f:
                    summary = json.load(f)
            else:
                summary = {
                    "status": "no_actual_data",
                    "message": "Using simulated data",
                }

            return summary

        except Exception as e:
            print(f"⚠️ Error extracting actual results: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            os.chdir(original_cwd)

    def _generate_publication_materials(self) -> Dict[str, Any]:
        """Generate publication-ready materials."""
        print("📋 Generating publication materials...")

        try:
            generator = PublicationResultsGenerator()
            generator.output_dir = self.publication_dir
            generator.generate_all_results()

            # Count generated files
            tables_count = len(list((self.publication_dir / "tables").glob("*.tex")))
            figures_count = len(list((self.publication_dir / "figures").glob("*.pdf")))
            data_count = len(list((self.publication_dir / "data").glob("*")))

            return {
                "status": "success",
                "tables_generated": tables_count,
                "figures_generated": figures_count,
                "data_files_generated": data_count,
            }

        except Exception as e:
            print(f"⚠️ Error generating publication materials: {e}")
            return {"status": "error", "message": str(e)}

    def _create_missing_components(self) -> Dict[str, Any]:
        """Create missing analysis components."""
        print("🔬 Creating missing analysis components...")

        try:
            generator = MissingAnalysisGenerator()
            generator.output_dir = self.missing_analysis_dir
            generator.generate_all_missing_components()

            # Count generated files
            figures_count = len(
                list((self.missing_analysis_dir / "figures").glob("*.pdf"))
            )
            data_count = len(list((self.missing_analysis_dir / "data").glob("*")))

            return {
                "status": "success",
                "additional_figures": figures_count,
                "additional_data": data_count,
            }

        except Exception as e:
            print(f"⚠️ Error creating missing components: {e}")
            return {"status": "error", "message": str(e)}

    def _validate_completeness(self) -> Dict[str, Any]:
        """Validate completeness of generated results."""
        print("✅ Validating completeness...")

        validation_report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "required_tables": {},
            "required_figures": {},
            "data_completeness": {},
            "missing_items": [],
            "warnings": [],
            "overall_status": "unknown",
        }

        # Required tables for thesis
        required_tables = [
            "run_matrix_table.tex",
            "main_performance_table.tex",
            "ablation_table.tex",
        ]

        # Required figures for thesis
        required_figures = [
            "roc_tss_curves.pdf",
            "reliability_diagrams.pdf",
            "cost_loss_analysis.pdf",
            "attention_heatmaps.pdf",
            "prospective_case_study.pdf",
            "ui_dashboard.pdf",
            "environmental_analysis.pdf",
            "cost_benefit_analysis.pdf",
            "architecture_evolution.pdf",
        ]

        # Check tables
        tables_dir = self.publication_dir / "tables"
        for table in required_tables:
            table_path = tables_dir / table
            validation_report["required_tables"][table] = {
                "exists": table_path.exists(),
                "size_bytes": table_path.stat().st_size if table_path.exists() else 0,
            }

            if not table_path.exists():
                validation_report["missing_items"].append(f"Table: {table}")

        # Check figures (both publication and missing analysis)
        figures_dirs = [
            self.publication_dir / "figures",
            self.missing_analysis_dir / "figures",
        ]

        for figure in required_figures:
            found = False
            for figures_dir in figures_dirs:
                figure_path = figures_dir / figure
                if figure_path.exists():
                    found = True
                    validation_report["required_figures"][figure] = {
                        "exists": True,
                        "location": str(figures_dir),
                        "size_bytes": figure_path.stat().st_size,
                    }
                    break

            if not found:
                validation_report["required_figures"][figure] = {"exists": False}
                validation_report["missing_items"].append(f"Figure: {figure}")

        # Check data completeness
        data_dirs = [
            self.publication_dir / "data",
            self.missing_analysis_dir / "data",
            self.actual_results_dir,
        ]

        required_data = [
            "main_performance_data.csv",
            "baseline_comparison.csv",
            "significance_tests.json",
            "environmental_impact.csv",
            "cost_benefit_analysis.csv",
        ]

        for data_file in required_data:
            found = False
            for data_dir in data_dirs:
                data_path = data_dir / data_file
                if data_path.exists():
                    found = True
                    validation_report["data_completeness"][data_file] = {
                        "exists": True,
                        "location": str(data_dir),
                        "size_bytes": data_path.stat().st_size,
                    }
                    break

            if not found:
                validation_report["data_completeness"][data_file] = {"exists": False}
                validation_report["missing_items"].append(f"Data: {data_file}")

        # Determine overall status
        missing_count = len(validation_report["missing_items"])
        if missing_count == 0:
            validation_report["overall_status"] = "complete"
        elif missing_count <= 3:
            validation_report["overall_status"] = "mostly_complete"
            validation_report["warnings"].append(f"{missing_count} items missing")
        else:
            validation_report["overall_status"] = "incomplete"
            validation_report["warnings"].append(
                f"{missing_count} items missing - significant gaps"
            )

        # Save validation report
        with open(self.validation_dir / "completeness_report.json", "w") as f:
            json.dump(validation_report, f, indent=2)

        return validation_report

    def _generate_final_summary(
        self,
        actual_data: Dict,
        publication_results: Dict,
        missing_components: Dict,
        validation_report: Dict,
    ) -> Dict[str, Any]:
        """Generate final comprehensive summary."""
        print("📋 Generating final summary...")

        final_summary = {
            "generation_timestamp": pd.Timestamp.now().isoformat(),
            "output_directory": str(self.output_dir),
            "steps_completed": {
                "actual_data_extraction": actual_data.get("status", "unknown"),
                "publication_materials": publication_results.get("status", "unknown"),
                "missing_components": missing_components.get("status", "unknown"),
                "validation": validation_report.get("overall_status", "unknown"),
            },
            "statistics": {
                "total_tables": publication_results.get("tables_generated", 0),
                "total_figures": (
                    publication_results.get("figures_generated", 0)
                    + missing_components.get("additional_figures", 0)
                ),
                "total_data_files": (
                    publication_results.get("data_files_generated", 0)
                    + missing_components.get("additional_data", 0)
                ),
                "missing_items": len(validation_report.get("missing_items", [])),
                "warnings": len(validation_report.get("warnings", [])),
            },
            "thesis_readiness": self._assess_thesis_readiness(validation_report),
            "next_steps": self._generate_next_steps(validation_report),
            "file_locations": {
                "tables": str(self.publication_dir / "tables"),
                "figures": [
                    str(self.publication_dir / "figures"),
                    str(self.missing_analysis_dir / "figures"),
                ],
                "data": [
                    str(self.publication_dir / "data"),
                    str(self.missing_analysis_dir / "data"),
                    str(self.actual_results_dir),
                ],
                "validation": str(self.validation_dir),
            },
        }

        # Save final summary
        with open(self.output_dir / "final_summary.json", "w") as f:
            json.dump(final_summary, f, indent=2)

        return final_summary

    def _assess_thesis_readiness(self, validation_report: Dict) -> Dict[str, Any]:
        """Assess readiness for thesis submission."""
        status = validation_report.get("overall_status", "unknown")
        missing_count = len(validation_report.get("missing_items", []))

        if status == "complete":
            readiness = {
                "status": "ready",
                "confidence": "high",
                "message": "All required components generated successfully",
            }
        elif status == "mostly_complete":
            readiness = {
                "status": "nearly_ready",
                "confidence": "medium",
                "message": f"Minor gaps ({missing_count} items) - can proceed with thesis",
            }
        else:
            readiness = {
                "status": "not_ready",
                "confidence": "low",
                "message": f"Significant gaps ({missing_count} items) - need to address missing components",
            }

        return readiness

    def _generate_next_steps(self, validation_report: Dict) -> List[str]:
        """Generate recommended next steps."""
        next_steps = []

        missing_items = validation_report.get("missing_items", [])

        if not missing_items:
            next_steps = [
                "✅ All components generated successfully",
                "📝 Review generated tables and figures for accuracy",
                "📊 Replace any simulated data with actual experimental results",
                "📖 Integrate results into thesis document",
                "🔍 Perform final quality check before submission",
            ]
        else:
            next_steps = [
                f"⚠️ Address {len(missing_items)} missing components:",
            ]

            for item in missing_items[:5]:  # Show first 5
                next_steps.append(f"   - {item}")

            if len(missing_items) > 5:
                next_steps.append(f"   - ... and {len(missing_items) - 5} more")

            next_steps.extend(
                [
                    "🔄 Re-run generation after addressing missing data",
                    "📊 Verify all experimental data is available",
                    "🔍 Check cluster job completion status",
                ]
            )

        return next_steps

    def _print_final_report(self, final_summary: Dict):
        """Print final report to console."""
        print("\n" + "=" * 60)
        print("FINAL THESIS RESULTS REPORT")
        print("=" * 60)

        # Status overview
        print(f"\n📊 GENERATION SUMMARY:")
        stats = final_summary["statistics"]
        print(f"   Tables generated: {stats['total_tables']}")
        print(f"   Figures generated: {stats['total_figures']}")
        print(f"   Data files created: {stats['total_data_files']}")
        print(f"   Missing items: {stats['missing_items']}")
        print(f"   Warnings: {stats['warnings']}")

        # Thesis readiness
        readiness = final_summary["thesis_readiness"]
        print(f"\n🎓 THESIS READINESS:")
        print(f"   Status: {readiness['status'].upper()}")
        print(f"   Confidence: {readiness['confidence'].upper()}")
        print(f"   Message: {readiness['message']}")

        # File locations
        print(f"\n📁 OUTPUT LOCATIONS:")
        locations = final_summary["file_locations"]
        print(f"   Tables: {locations['tables']}")
        print(f"   Figures: {locations['figures'][0]}")
        print(f"            {locations['figures'][1]}")
        print(f"   Data: {locations['data'][0]}")
        print(f"         {locations['data'][1]}")
        print(f"   Validation: {locations['validation']}")

        # Next steps
        print(f"\n📋 NEXT STEPS:")
        for step in final_summary["next_steps"]:
            print(f"   {step}")

        print(f"\n📄 Full report saved to: {self.output_dir / 'final_summary.json'}")
        print("=" * 60)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate all results needed for EVEREST thesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_thesis_results.py                    # Full generation
    python generate_thesis_results.py --mode extract     # Extract actual data only
    python generate_thesis_results.py --mode publish     # Publication materials only
    python generate_thesis_results.py --mode missing     # Missing components only
    python generate_thesis_results.py --mode validate    # Validation only
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["full", "extract", "publish", "missing", "validate"],
        default="full",
        help="Generation mode (default: full)",
    )

    parser.add_argument(
        "--output-dir",
        default="thesis_results",
        help="Output directory (default: thesis_results)",
    )

    args = parser.parse_args()

    # Import pandas here to avoid import issues
    import pandas as pd

    # Create orchestrator
    orchestrator = ThesisResultsOrchestrator(args.output_dir)

    try:
        if args.mode == "full":
            orchestrator.generate_all_results()
        elif args.mode == "extract":
            orchestrator._extract_actual_results()
        elif args.mode == "publish":
            orchestrator._generate_publication_materials()
        elif args.mode == "missing":
            orchestrator._create_missing_components()
        elif args.mode == "validate":
            validation_report = orchestrator._validate_completeness()
            print(
                f"\nValidation complete. Status: {validation_report['overall_status']}"
            )
            if validation_report["missing_items"]:
                print(f"Missing items: {validation_report['missing_items']}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
