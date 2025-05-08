#!/usr/bin/env python3
"""
Script to generate the SolarKnowledge_Training_No_Widgets.ipynb file with direct code execution
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
                "# SolarKnowledge Model Training - PyTorch Implementation\n",
                "\n",
                "This notebook provides an interface to train SolarKnowledge models for solar flare prediction using the enhanced PyTorch implementation with batch normalization, AdamW optimizer, and cosine annealing learning rate scheduling."
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
                "!pip install matplotlib seaborn tqdm"
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
                "# Import run_all_trainings function\n",
                "try:\n",
                "    from models.SolarKnowledge_run_all_trainings_pytorch import train as run_training, set_seed\n",
                "    from models.SolarKnowledge_model_pytorch import SolarKnowledge\n",
                "    from models.utils import supported_flare_class\n",
                "    \n",
                "    # Ensure consistent seed setting\n",
                "    set_seed(RANDOM_SEED)\n",
                "    \n",
                "    print(\"Successfully imported training modules\")\n",
                "except ImportError as e:\n",
                "    print(f\"Error importing modules: {e}\")\n",
                "    print(\"\\nPlease ensure the following files exist in the 'models' directory:\")\n",
                "    print(\"- SolarKnowledge_model_pytorch.py\")\n",
                "    print(\"- SolarKnowledge_run_all_trainings_pytorch.py\")\n",
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
                "# Model hyperparameters\n",
                "epochs = 300\n",
                "patience = 10\n",
                "learning_rate = 5e-5  # Options: 1e-5, 5e-5, 1e-4, 5e-4\n",
                "scheduler_type = 'cosine_with_restarts'  # Options: 'cosine_with_restarts', 'cosine_annealing', 'reduce_on_plateau'\n",
                "embed_dim = 256  # Options: 128, 256, 512\n",
                "transformer_blocks = 8  # Options: 6, 8, 12\n",
                "batch_size = 512  # Options: 256, 512, 1024\n",
                "use_batch_norm = True\n",
                "use_focal_loss = True\n",
                "compare_models = False  # Whether to compare models after training\n",
                "\n",
                "# Configure scheduler parameters based on selected scheduler\n",
                "if scheduler_type == 'cosine_with_restarts':\n",
                "    scheduler_params = {\n",
                "        \"T_0\": 10,         # Initial cycle length\n",
                "        \"T_mult\": 2,       # Cycle length multiplier\n",
                "        \"min_lr\": 1e-7     # Minimum learning rate\n",
                "    }\n",
                "elif scheduler_type == 'cosine_annealing':\n",
                "    scheduler_params = {\n",
                "        \"T_max\": 10,       # Cycle length\n",
                "        \"min_lr\": 1e-7     # Minimum learning rate\n",
                "    }\n",
                "elif scheduler_type == 'reduce_on_plateau':\n",
                "    scheduler_params = {\n",
                "        \"factor\": 0.2,     # Reduction factor\n",
                "        \"patience\": 5,     # Patience\n",
                "        \"min_lr\": 1e-7     # Minimum learning rate\n",
                "    }\n",
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
                "print(f\"Scheduler: {scheduler_type}\")\n",
                "print(f\"Embedding Dimension: {embed_dim}\")\n",
                "print(f\"Transformer Blocks: {transformer_blocks}\")\n",
                "print(f\"Batch Size: {batch_size}\")\n",
                "print(f\"Using Batch Normalization: {'Yes' if use_batch_norm else 'No'}\")\n",
                "print(f\"Using Focal Loss: {'Yes' if use_focal_loss else 'No'}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Run Training\n",
                "\n",
                "Execute the training with the configured parameters. You can modify the parameters in the cell above and re-run this cell as needed."
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
                "                    # Create a customized model instance with our parameters\n",
                "                    model = SolarKnowledge(early_stopping_patience=patience)\n",
                "                    \n",
                "                    # Set up custom hyperparameters dictionary for model_tracking\n",
                "                    hyperparams = {\n",
                "                        \"learning_rate\": learning_rate,\n",
                "                        \"weight_decay\": 1e-4,\n",
                "                        \"batch_size\": batch_size,\n",
                "                        \"early_stopping_patience\": patience,\n",
                "                        \"epochs\": epochs,\n",
                "                        \"num_transformer_blocks\": transformer_blocks,\n",
                "                        \"embed_dim\": embed_dim,\n",
                "                        \"num_heads\": 8,  # Could be made configurable too\n",
                "                        \"ff_dim\": embed_dim * 2,  # typically 2x embed_dim\n",
                "                        \"dropout_rate\": 0.2,\n",
                "                        \"focal_loss\": use_focal_loss,\n",
                "                        \"focal_loss_alpha\": 0.25,\n",
                "                        \"focal_loss_gamma\": 2.0,\n",
                "                        \"framework\": \"pytorch\",\n",
                "                        \"weight_initialization\": \"tf_compatible\",\n",
                "                        \"gradient_clipping\": True,\n",
                "                        \"max_grad_norm\": 1.0,\n",
                "                        \"scheduler\": scheduler_type,\n",
                "                        \"scheduler_params\": scheduler_params,\n",
                "                        \"use_batch_norm\": use_batch_norm,\n",
                "                        \"optimizer\": \"AdamW\",\n",
                "                        \"random_seed\": RANDOM_SEED,\n",
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
                "                        scheduler_type=scheduler_type,\n",
                "                        scheduler_params=scheduler_params,\n",
                "                        batch_size=batch_size,\n",
                "                        learning_rate=learning_rate,\n",
                "                        embed_dim=embed_dim,\n",
                "                        transformer_blocks=transformer_blocks,\n",
                "                        use_batch_norm=use_batch_norm,\n",
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
                "## Compare Models (Optional)\n",
                "\n",
                "Run this cell if you want to compare the models after training"
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