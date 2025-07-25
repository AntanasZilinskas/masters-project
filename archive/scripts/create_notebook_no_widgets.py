#!/usr/bin/env python3
"""
Script to generate the SolarKnowledge_Training.ipynb file with direct code execution
including both training and testing sequences with visualizations
"""
import json
import os

# Define the notebook content
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# SolarKnowledge Model Training & Testing - PyTorch Implementation\n",
                "\n",
                "This notebook provides an interface to train and test SolarKnowledge models for solar flare prediction using the PyTorch implementation that exactly matches the TensorFlow version's behavior for direct comparison."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Setup & Import Dependencies"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install required packages if not already installed\n",
                "!pip install matplotlib seaborn tqdm scikit-learn"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "import os\n",
                "import random\n",
                "import numpy as np\n",
                "import json\n",
                "from datetime import datetime\n",
                "\n",
                "# Ensure models directory is in the path\n",
                "models_dir = \"models\"\n",
                "if not os.path.exists(models_dir):\n",
                "    os.makedirs(models_dir)\n",
                "if models_dir not in sys.path:\n",
                "    sys.path.append(models_dir)\n",
                "\n",
                "# Import necessary modules\n",
                "import torch\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Set random seed for reproducibility\n",
                "RANDOM_SEED = 42\n",
                "random.seed(RANDOM_SEED)\n",
                "np.random.seed(RANDOM_SEED)\n",
                "torch.manual_seed(RANDOM_SEED)\n",
                "if torch.cuda.is_available():\n",
                "    torch.cuda.manual_seed(RANDOM_SEED)\n",
                "    torch.cuda.manual_seed_all(RANDOM_SEED)\n",
                "torch.backends.cudnn.deterministic = True\n",
                "torch.backends.cudnn.benchmark = False\n",
                "\n",
                "# Check if PyTorch is using GPU\n",
                "print(f\"PyTorch version: {torch.__version__}\")\n",
                "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"CUDA device: {torch.cuda.get_device_name(0)}\")\n",
                "elif hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available():\n",
                "    print(\"Apple Silicon MPS available\")\n",
                "else:\n",
                "    print(\"Using CPU\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import required model functions\n",
                "try:\n",
                "    # Training modules\n",
                "    from models.SolarKnowledge_run_all_trainings_pytorch import train as run_training, set_seed\n",
                "    from models.SolarKnowledge_model_pytorch import SolarKnowledge\n",
                "    from models.utils import supported_flare_class, get_training_data\n",
                "    \n",
                "    # Testing modules\n",
                "    from models.SolarKnowledge_run_all_tests_pytorch import test_model, find_latest_model_version\n",
                "    from models.utils import get_testing_data\n",
                "    from sklearn.metrics import classification_report, confusion_matrix\n",
                "    \n",
                "    # Ensure consistent seed setting\n",
                "    set_seed(RANDOM_SEED)\n",
                "    \n",
                "    print(\"Successfully imported all modules\")\n",
                "except ImportError as e:\n",
                "    print(f\"Error importing modules: {e}\")\n",
                "    print(\"\\nPlease ensure the following files exist in the 'models' directory:\")\n",
                "    print(\"- SolarKnowledge_model_pytorch.py\")\n",
                "    print(\"- SolarKnowledge_run_all_trainings_pytorch.py\")\n",
                "    print(\"- SolarKnowledge_run_all_tests_pytorch.py\")\n",
                "    print(\"- utils.py\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Configure Training Parameters\n",
                "\n",
                "Edit the parameters below to configure your training settings."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Configuration parameters (modify these as needed)\n",
                "\n",
                "# Basic parameters\n",
                "flare_classes = ['C']  # Options: 'C', 'M', 'M5' - you can select multiple, e.g. ['C', 'M', 'M5']\n",
                "time_windows = ['24']  # Options: '24', '48', '72' - you can select multiple\n",
                "version = None  # Set to None for auto version or specify (e.g., '2')\n",
                "description = None  # Optional description of this training run\n",
                "auto_increment = True  # Whether to auto-increment version numbers\n",
                "\n",
                "# Model hyperparameters - TensorFlow equivalent values\n",
                "epochs = 100  # TensorFlow models use 100 epochs\n",
                "patience = 15  # Increased to allow more convergence time\n",
                "learning_rate = 1e-4  # Match TensorFlow's exact default\n",
                "batch_size = 512  # Match TensorFlow's default\n",
                "embed_dim = 128  # Match TensorFlow's default\n",
                "transformer_blocks = 6  # Match TensorFlow's default\n",
                "use_focal_loss = True  # Match TensorFlow model\n",
                "compare_models = False  # Whether to compare models after training\n",
                "\n",
                "# Configure optimizer and regularization to match TensorFlow\n",
                "optimizer_type = 'Adam'  # TensorFlow uses standard Adam, not AdamW\n",
                "l1_regularization = 1e-5  # L1 regularization factor (match TensorFlow)\n",
                "l2_regularization = 1e-4  # L2 regularization factor (match TensorFlow)\n",
                "\n",
                "# Configure LR scheduler to match TensorFlow's ReduceLROnPlateau\n",
                "scheduler_params = {\n",
                "    \"monitor\": \"loss\",     # Match TensorFlow (monitors training loss)\n",
                "    \"factor\": 0.5,          # Match TensorFlow (halves the learning rate)\n",
                "    \"patience\": 3,          # Match TensorFlow (waits 3 epochs)\n",
                "    \"min_lr\": 1e-6          # Match TensorFlow (minimum learning rate)\n",
                "    # Note: TensorFlow accepts 'verbose' parameter but PyTorch does not\n",
                "}\n",
                "\n",
                "# Display current settings\n",
                "print(\"\\nTraining Configuration:\")\n",
                "print(f\"Flare Classes: {', '.join(flare_classes)}\")\n",
                "print(f\"Time Windows: {', '.join(time_windows)}\")\n",
                "print(f\"Version: {'Auto' if version is None else version}\")\n",
                "print(f\"Description: {'None' if description is None else description}\")\n",
                "print(f\"Auto-increment: {'Yes' if auto_increment else 'No'}\")\n",
                "\n",
                "print(\"\\nHyperparameters:\")\n",
                "print(f\"Epochs: {epochs}\")\n",
                "print(f\"Early Stopping Patience: {patience}\")\n",
                "print(f\"Learning Rate: {learning_rate}\")\n",
                "print(f\"Optimizer: {optimizer_type}\")\n",
                "print(f\"L1 Regularization: {l1_regularization}\")\n",
                "print(f\"L2 Regularization: {l2_regularization}\")\n",
                "print(f\"Scheduler: ReduceLROnPlateau ({scheduler_params['factor']} factor, {scheduler_params['patience']} patience)\")\n",
                "print(f\"Embedding Dimension: {embed_dim}\")\n",
                "print(f\"Transformer Blocks: {transformer_blocks}\")\n",
                "print(f\"Batch Size: {batch_size}\")\n",
                "print(f\"Using Focal Loss: {'Yes' if use_focal_loss else 'No'}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Run Training\n",
                "\n",
                "Execute the training with the configured parameters. You can monitor progress with epoch-by-epoch metrics."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Store trained models information\n",
                "trained_models = []\n",
                "versions = []\n",
                "\n",
                "# Validate selections\n",
                "if not flare_classes:\n",
                "    print(\"Error: Please select at least one flare class\")\n",
                "else:\n",
                "    if not time_windows:\n",
                "        print(\"Error: Please select at least one time window\")\n",
                "    else:\n",
                "        print(\"\\nStarting Training...\")\n",
                "        print(\"-\"*50)\n",
                "        \n",
                "        # Run training for each combination\n",
                "        for time_window in time_windows:\n",
                "            for flare_class in flare_classes:\n",
                "                if flare_class not in supported_flare_class:\n",
                "                    print(f\"Unsupported flare class: {flare_class}\")\n",
                "                    continue\n",
                "                    \n",
                "                print(f\"\\nTraining model for {flare_class}-class flares with {time_window}h window\")\n",
                "                \n",
                "                try:\n",
                "                    # First, get the input shape from the training data\n",
                "                    X_train, _ = get_training_data(time_window, flare_class)\n",
                "                    input_shape = (X_train.shape[1], X_train.shape[2])\n",
                "                    print(f\"Input shape: {input_shape}\")\n",
                "                    \n",
                "                    # Create a customized model instance with our parameters\n",
                "                    model = SolarKnowledge(early_stopping_patience=patience)\n",
                "                    \n",
                "                    # Build the model\n",
                "                    model.build_base_model(\n",
                "                        input_shape=input_shape,\n",
                "                        embed_dim=embed_dim,\n",
                "                        num_heads=4,  # Match TensorFlow\n",
                "                        ff_dim=256,  # Match TensorFlow\n",
                "                        num_transformer_blocks=transformer_blocks,\n",
                "                        dropout_rate=0.2\n",
                "                    )\n",
                "                    \n",
                "                    # Compile the model\n",
                "                    model.compile(\n",
                "                        use_focal_loss=use_focal_loss,\n",
                "                        learning_rate=learning_rate\n",
                "                    )\n",
                "                    \n",
                "                    # Model.model is the actual PyTorch model\n",
                "                    if hasattr(model.model, 'l1_regularizer') and hasattr(model.model, 'l2_regularizer'):\n",
                "                        # Set regularization strengths to match TensorFlow\n",
                "                        model.model.l1_regularizer = l1_regularization\n",
                "                        model.model.l2_regularizer = l2_regularization\n",
                "                        print(f\"Set regularization: L1={l1_regularization}, L2={l2_regularization}\")\n",
                "                    \n",
                "                    # Set up custom hyperparameters dictionary for model_tracking\n",
                "                    hyperparams = {\n",
                "                        \"learning_rate\": learning_rate,\n",
                "                        \"weight_decay\": 0.0,        # Match TensorFlow (no weight decay)\n",
                "                        \"batch_size\": batch_size,\n",
                "                        \"early_stopping_patience\": patience,\n",
                "                        \"early_stopping_metric\": \"loss\",  # Match TensorFlow\n",
                "                        \"epochs\": epochs,\n",
                "                        \"num_transformer_blocks\": transformer_blocks,\n",
                "                        \"embed_dim\": embed_dim,\n",
                "                        \"num_heads\": 4,  # Match TensorFlow\n",
                "                        \"ff_dim\": 256,  # Match TensorFlow\n",
                "                        \"dropout_rate\": 0.2,\n",
                "                        \"focal_loss\": use_focal_loss,\n",
                "                        \"focal_loss_alpha\": 0.25,\n",
                "                        \"focal_loss_gamma\": 2.0,\n",
                "                        \"framework\": \"pytorch\",\n",
                "                        \"gradient_clipping\": True,\n",
                "                        \"max_grad_norm\": 1.0,\n",
                "                        \"input_shape\": input_shape,\n",
                "                        \"lr_scheduler\": {\n",
                "                            \"type\": \"ReduceLROnPlateau\",\n",
                "                            \"monitor\": scheduler_params[\"monitor\"],\n",
                "                            \"factor\": scheduler_params[\"factor\"],\n",
                "                            \"patience\": scheduler_params[\"patience\"],\n",
                "                            \"min_lr\": scheduler_params[\"min_lr\"]\n",
                "                        },\n",
                "                        \"regularization\": {\n",
                "                            \"l1\": l1_regularization,\n",
                "                            \"l2\": l2_regularization\n",
                "                        },\n",
                "                        \"optimizer\": optimizer_type,\n",
                "                        \"random_seed\": RANDOM_SEED\n",
                "                    }\n",
                "                    \n",
                "                    # Run the training function\n",
                "                    model_dir, trained_version = run_training(\n",
                "                        time_window,\n",
                "                        flare_class,\n",
                "                        version=version,\n",
                "                        description=description,\n",
                "                        auto_increment=auto_increment,\n",
                "                        # Pass custom parameters to override defaults\n",
                "                        custom_model=model,\n",
                "                        custom_hyperparams=hyperparams,\n",
                "                        epochs=epochs,\n",
                "                        batch_size=batch_size,\n",
                "                        learning_rate=learning_rate,\n",
                "                        embed_dim=embed_dim,\n",
                "                        transformer_blocks=transformer_blocks,\n",
                "                        use_focal_loss=use_focal_loss\n",
                "                    )\n",
                "                    \n",
                "                    trained_models.append({\n",
                "                        \"time_window\": time_window,\n",
                "                        \"flare_class\": flare_class,\n",
                "                        \"model_dir\": model_dir,\n",
                "                        \"version\": trained_version\n",
                "                    })\n",
                "                    versions.append(trained_version)\n",
                "                    \n",
                "                    print(f\"\\nModel saved to {model_dir}\")\n",
                "                    print(\"-\"*50)\n",
                "                    \n",
                "                except Exception as e:\n",
                "                    print(f\"Error training model: {e}\")\n",
                "                    import traceback\n",
                "                    traceback.print_exc()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Configure Testing Parameters\n",
                "\n",
                "Edit the parameters below to configure your testing settings."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Testing configuration\n",
                "test_flare_classes = flare_classes  # Default to same as training\n",
                "test_time_windows = time_windows    # Default to same as training\n",
                "mc_passes = 30                      # Number of Monte Carlo dropout passes for uncertainty estimation\n",
                "plot_uncertainties = True           # Whether to generate uncertainty visualization plots\n",
                "test_latest = True                  # Test the latest model versions by default\n",
                "test_specific_version = None        # Set to a specific version string to test it instead\n",
                "\n",
                "# Display testing settings\n",
                "print(\"\\nTesting Configuration:\")\n",
                "print(f\"Flare Classes: {', '.join(test_flare_classes)}\")\n",
                "print(f\"Time Windows: {', '.join(test_time_windows)}\")\n",
                "print(f\"Monte Carlo Passes: {mc_passes}\")\n",
                "print(f\"Generate Uncertainty Plots: {'Yes' if plot_uncertainties else 'No'}\")\n",
                "print(f\"Test Latest Models: {'Yes' if test_latest else 'No'}\")\n",
                "print(f\"Test Specific Version: {test_specific_version if test_specific_version else 'No'}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Run Testing\n",
                "\n",
                "Execute testing on the trained models with the configured parameters. This will provide detailed performance metrics and uncertainty visualizations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Dictionary to store all test results\n",
                "all_test_results = {}\n",
                "\n",
                "# Run tests for each model\n",
                "for time_window in test_time_windows:\n",
                "    if time_window not in all_test_results:\n",
                "        all_test_results[time_window] = {}\n",
                "        \n",
                "    for flare_class in test_flare_classes:\n",
                "        if flare_class not in supported_flare_class:\n",
                "            print(f\"Unsupported flare class: {flare_class}\")\n",
                "            continue\n",
                "            \n",
                "        print(f\"\\nTesting model for {flare_class}-class flares with {time_window}h window\")\n",
                "        print(\"-\"*50)\n",
                "        \n",
                "        try:\n",
                "            # Find the model directory to test\n",
                "            model_dir = None\n",
                "            \n",
                "            if test_specific_version:\n",
                "                # Look for a specific version\n",
                "                model_patterns = [\n",
                "                    # New structure: models/models/SolarKnowledge-v*\n",
                "                    os.path.join(\"models\", \"models\", f\"SolarKnowledge-v{test_specific_version}-{flare_class}-{time_window}h\"),\n",
                "                    # Old structure: models/SolarKnowledge-v*\n",
                "                    os.path.join(\"models\", f\"SolarKnowledge-v{test_specific_version}-{flare_class}-{time_window}h\"),\n",
                "                ]\n",
                "                \n",
                "                for pattern in model_patterns:\n",
                "                    if os.path.exists(pattern):\n",
                "                        model_dir = pattern\n",
                "                        break\n",
                "                        \n",
                "                if not model_dir:\n",
                "                    print(f\"Could not find model with version {test_specific_version}\")\n",
                "                    continue\n",
                "                    \n",
                "            else:\n",
                "                # Find the latest model version\n",
                "                model_dir = find_latest_model_version(flare_class, time_window)\n",
                "                \n",
                "                if not model_dir:\n",
                "                    print(f\"Could not find a model for {flare_class}-class flares with {time_window}h window\")\n",
                "                    continue\n",
                "            \n",
                "            print(f\"Testing model at: {model_dir}\")\n",
                "            \n",
                "            # Run the test\n",
                "            test_results = test_model(\n",
                "                time_window,\n",
                "                flare_class,\n",
                "                use_latest=True,\n",
                "                mc_passes=mc_passes,\n",
                "                plot_uncertainties=plot_uncertainties\n",
                "            )\n",
                "            \n",
                "            # Store results\n",
                "            if test_results is not None:\n",
                "                all_test_results[time_window][flare_class] = test_results\n",
                "                \n",
                "                # Print a summary of the results\n",
                "                print(\"\\nTest Results Summary:\")\n",
                "                print(f\"Accuracy: {test_results['accuracy']}\")\n",
                "                print(f\"TSS: {test_results['TSS']}\")\n",
                "                print(f\"Precision: {test_results['precision']}\")\n",
                "                print(f\"Recall: {test_results['recall']}\")\n",
                "                print(f\"Balanced Accuracy: {test_results['balanced_accuracy']}\")\n",
                "                \n",
                "                # Show uncertainty information if available\n",
                "                if 'mean_uncertainty' in test_results:\n",
                "                    print(f\"\\nUncertainty Metrics:\")\n",
                "                    print(f\"Mean Uncertainty: {test_results['mean_uncertainty']:.4f}\")\n",
                "                    print(f\"Mean Confidence: {test_results['mean_confidence']:.4f}\")\n",
                "                    print(f\"Mean Entropy: {test_results['mean_entropy']:.4f}\")\n",
                "                \n",
                "                # Print confusion matrix if available\n",
                "                if 'confusion_matrix' in test_results:\n",
                "                    print(\"\\nConfusion Matrix:\")\n",
                "                    cm = test_results['confusion_matrix']\n",
                "                    print(f\"[[{cm[0][0]}, {cm[0][1]}]\")\n",
                "                    print(f\" [{cm[1][0]}, {cm[1][1]}]]\")\n",
                "                    \n",
                "                    # Calculate and print additional metrics\n",
                "                    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]\n",
                "                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0\n",
                "                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0\n",
                "                    print(f\"Sensitivity (TPR): {sensitivity:.4f}\")\n",
                "                    print(f\"Specificity (TNR): {specificity:.4f}\")\n",
                "                    print(f\"False Positive Rate: {1-specificity:.4f}\")\n",
                "                    print(f\"False Negative Rate: {1-sensitivity:.4f}\")\n",
                "                \n",
                "                # Show sample distribution\n",
                "                if all(k in test_results for k in [\"test_samples\", \"positive_samples\", \"negative_samples\"]):\n",
                "                    print(f\"\\nTest Data Distribution:\")\n",
                "                    pos = test_results[\"positive_samples\"]\n",
                "                    neg = test_results[\"negative_samples\"]\n",
                "                    total = test_results[\"test_samples\"]\n",
                "                    print(f\"Total samples: {total}\")\n",
                "                    print(f\"Positive samples: {pos} ({pos/total:.2%})\")\n",
                "                    print(f\"Negative samples: {neg} ({neg/total:.2%})\")\n",
                "            else:\n",
                "                print(\"No test results returned.\")\n",
                "                \n",
                "        except Exception as e:\n",
                "            print(f\"Error testing model: {e}\")\n",
                "            import traceback\n",
                "            traceback.print_exc()\n",
                "            \n",
                "        print(\"-\"*50)\n",
                "        \n",
                "# Save all test results to a JSON file\n",
                "results_filename = f\"pytorch_test_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.json\"\n",
                "with open(results_filename, \"w\") as f:\n",
                "    json.dump(all_test_results, f, indent=4)\n",
                "    \n",
                "print(f\"\\nAll test results saved to {results_filename}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Visualize Test Results\n",
                "\n",
                "Create visualizations to better understand model performance."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualize TSS values across different flare classes and time windows\n",
                "if all_test_results:\n",
                "    plt.figure(figsize=(12, 6))\n",
                "    \n",
                "    # Extract TSS values\n",
                "    x_labels = []\n",
                "    tss_values = []\n",
                "    colors = {'C': 'blue', 'M': 'green', 'M5': 'red'}\n",
                "    \n",
                "    for tw in sorted(all_test_results.keys()):\n",
                "        for fc in sorted(all_test_results[tw].keys()):\n",
                "            if 'TSS' in all_test_results[tw][fc]:\n",
                "                x_labels.append(f\"{fc}-{tw}h\")\n",
                "                tss_values.append(all_test_results[tw][fc]['TSS'])\n",
                "    \n",
                "    # Create bar chart\n",
                "    bars = plt.bar(x_labels, tss_values)\n",
                "    \n",
                "    # Color bars based on flare class\n",
                "    for i, label in enumerate(x_labels):\n",
                "        fc = label.split('-')[0]\n",
                "        if fc in colors:\n",
                "            bars[i].set_color(colors[fc])\n",
                "    \n",
                "    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)\n",
                "    plt.title('TSS by Flare Class and Time Window')\n",
                "    plt.xlabel('Flare Class - Time Window')\n",
                "    plt.ylabel('True Skill Statistic (TSS)')\n",
                "    plt.ylim(0, 1.0)\n",
                "    plt.grid(axis='y', alpha=0.3)\n",
                "    \n",
                "    # Add a legend\n",
                "    from matplotlib.patches import Patch\n",
                "    legend_elements = [Patch(facecolor=color, label=fc) for fc, color in colors.items()]\n",
                "    plt.legend(handles=legend_elements, title='Flare Class')\n",
                "    \n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "    \n",
                "    # Create a comparison of key metrics\n",
                "    metrics = ['accuracy', 'TSS', 'precision', 'recall', 'balanced_accuracy']\n",
                "    plt.figure(figsize=(15, 10))\n",
                "    \n",
                "    # Plot different metrics side by side for each model\n",
                "    for i, metric in enumerate(metrics):\n",
                "        plt.subplot(len(metrics), 1, i+1)\n",
                "        \n",
                "        x_labels = []\n",
                "        values = []\n",
                "        \n",
                "        for tw in sorted(all_test_results.keys()):\n",
                "            for fc in sorted(all_test_results[tw].keys()):\n",
                "                if metric in all_test_results[tw][fc]:\n",
                "                    x_labels.append(f\"{fc}-{tw}h\")\n",
                "                    values.append(all_test_results[tw][fc][metric])\n",
                "        \n",
                "        # Create bar chart\n",
                "        bars = plt.bar(x_labels, values)\n",
                "        \n",
                "        # Color bars based on flare class\n",
                "        for j, label in enumerate(x_labels):\n",
                "            fc = label.split('-')[0]\n",
                "            if fc in colors:\n",
                "                bars[j].set_color(colors[fc])\n",
                "        \n",
                "        plt.title(f'{metric.replace(\"_\", \" \").title()}')\n",
                "        plt.ylim(0, 1.0)\n",
                "        plt.grid(axis='y', alpha=0.3)\n",
                "        \n",
                "        # Only show x-labels for the bottom subplot\n",
                "        if i == len(metrics)-1:\n",
                "            plt.xlabel('Flare Class - Time Window')\n",
                "    \n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "else:\n",
                "    print(\"No test results available for visualization\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Compare Results with TensorFlow Implementation\n",
                "\n",
                "If you have saved TensorFlow model results, you can load and compare them with the PyTorch implementation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check if TensorFlow results exist and compare\n",
                "tf_results_file = \"tensorflow_results.json\"  # Change this to your TensorFlow results file name\n",
                "\n",
                "if os.path.exists(tf_results_file):\n",
                "    try:\n",
                "        # Load TensorFlow results\n",
                "        with open(tf_results_file, 'r') as f:\n",
                "            tf_results = json.load(f)\n",
                "        \n",
                "        # Prepare for comparison\n",
                "        comparison_metrics = ['TSS', 'accuracy', 'precision', 'recall']\n",
                "        frameworks = ['TensorFlow', 'PyTorch']\n",
                "        \n",
                "        # For each time window and flare class, collect and compare metrics\n",
                "        for tw in sorted(all_test_results.keys()):\n",
                "            if tw in tf_results:\n",
                "                print(f\"\\nComparison for {tw}h Time Window:\")\n",
                "                print(\"-\"*50)\n",
                "                \n",
                "                # Print header row\n",
                "                header = \"Flare Class | Metric | \" + \" | \".join(frameworks)\n",
                "                print(header)\n",
                "                print(\"-\" * len(header))\n",
                "                \n",
                "                for fc in sorted(all_test_results[tw].keys()):\n",
                "                    if fc in tf_results[tw]:\n",
                "                        for metric in comparison_metrics:\n",
                "                            if metric in tf_results[tw][fc] and metric in all_test_results[tw][fc]:\n",
                "                                # Format values for printing\n",
                "                                tf_val = tf_results[tw][fc][metric]\n",
                "                                pt_val = all_test_results[tw][fc][metric]\n",
                "                                \n",
                "                                # Calculate difference\n",
                "                                if isinstance(tf_val, (int, float)) and isinstance(pt_val, (int, float)):\n",
                "                                    diff = pt_val - tf_val\n",
                "                                    diff_str = f\"{diff:+.4f}\"\n",
                "                                else:\n",
                "                                    diff_str = \"N/A\"\n",
                "                                \n",
                "                                # Print comparison row\n",
                "                                print(f\"{fc:^10} | {metric:^12} | {tf_val:^10} | {pt_val:^10} | {diff_str:^10}\")\n",
                "                        \n",
                "                        # Add a separator between flare classes\n",
                "                        print(\"-\" * len(header))\n",
                "    except Exception as e:\n",
                "        print(f\"Error comparing results: {e}\")\n",
                "else:\n",
                "    print(f\"TensorFlow results file '{tf_results_file}' not found. Skipping comparison.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Compare Models (Optional)\n",
                "\n",
                "Run this cell if you want to compare different model versions after training"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Compare models if requested and if models were trained\n",
                "if compare_models and trained_models:\n",
                "    print(\"\\nComparing trained models...\")\n",
                "    try:\n",
                "        from models.model_tracking import compare_models as compare_models_function\n",
                "        comparison = compare_models_function(\n",
                "            list(set(versions)),  # Unique versions\n",
                "            flare_classes,\n",
                "            time_windows\n",
                "        )\n",
                "        print(\"\\nModel Comparison:\")\n",
                "        print(comparison)\n",
                "    except Exception as e:\n",
                "        print(f\"Error comparing models: {e}\")\n",
                "else:\n",
                "    if not compare_models:\n",
                "        print(\"Model comparison skipped (set compare_models = True to enable)\")\n",
                "    elif not trained_models:\n",
                "        print(\"No models were trained to compare\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.8"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook to a file
with open('SolarKnowledge_Training.ipynb', 'w') as f:
    json.dump(notebook, f)

print("Created Jupyter notebook: SolarKnowledge_Training.ipynb")
print("Run this notebook with: jupyter notebook SolarKnowledge_Training.ipynb") 