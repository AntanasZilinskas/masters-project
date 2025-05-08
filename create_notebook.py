#!/usr/bin/env python3
"""
Script to generate the SolarKnowledge_Training.ipynb file
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
                "This notebook provides an interactive interface to train SolarKnowledge models for solar flare prediction using the PyTorch implementation."
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
                "!pip install ipywidgets matplotlib seaborn tqdm"
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
                "\n",
                "# Ensure models directory is in the path\n",
                "models_dir = \"models\"\n",
                "if not os.path.exists(models_dir):\n",
                "    os.makedirs(models_dir)\n",
                "if models_dir not in sys.path:\n",
                "    sys.path.append(models_dir)\n",
                "\n",
                "# Import necessary modules\n",
                "import ipywidgets as widgets\n",
                "from IPython.display import display, HTML\n",
                "import numpy as np\n",
                "import torch\n",
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
                "    from models.SolarKnowledge_run_all_trainings_pytorch import train as run_training\n",
                "    from models.SolarKnowledge_model_pytorch import SolarKnowledge\n",
                "    from models.utils import supported_flare_class\n",
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
                "## Create Interactive Controls"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Define controls for training parameters\n",
                "flare_dropdown = widgets.SelectMultiple(\n",
                "    options=['C', 'M', 'M5'],\n",
                "    value=['C'],\n",
                "    description='Flare Class:',\n",
                "    disabled=False\n",
                ")\n",
                "\n",
                "time_window_dropdown = widgets.SelectMultiple(\n",
                "    options=['24', '48', '72'],\n",
                "    value=['24'],\n",
                "    description='Time Window (h):',\n",
                "    disabled=False\n",
                ")\n",
                "\n",
                "version_text = widgets.Text(\n",
                "    value='',\n",
                "    placeholder='Auto',\n",
                "    description='Version:',\n",
                "    disabled=False\n",
                ")\n",
                "\n",
                "description_text = widgets.Text(\n",
                "    value='',\n",
                "    placeholder='Optional description',\n",
                "    description='Description:',\n",
                "    disabled=False\n",
                ")\n",
                "\n",
                "auto_increment_checkbox = widgets.Checkbox(\n",
                "    value=True,\n",
                "    description='Auto-increment version',\n",
                "    disabled=False\n",
                ")\n",
                "\n",
                "compare_checkbox = widgets.Checkbox(\n",
                "    value=False,\n",
                "    description='Compare models after training',\n",
                "    disabled=False\n",
                ")\n",
                "\n",
                "# Output area to display training progress\n",
                "output = widgets.Output()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Function to run the training with selected parameters\n",
                "def run_training_with_params(button):\n",
                "    with output:\n",
                "        output.clear_output()\n",
                "        \n",
                "        # Get selected parameters\n",
                "        flare_classes = flare_dropdown.value\n",
                "        time_windows = time_window_dropdown.value\n",
                "        version = version_text.value if version_text.value else None\n",
                "        description = description_text.value if description_text.value else None\n",
                "        auto_increment = auto_increment_checkbox.value\n",
                "        \n",
                "        # Validate selections\n",
                "        if not flare_classes:\n",
                "            print(\"Error: Please select at least one flare class\")\n",
                "            return\n",
                "        if not time_windows:\n",
                "            print(\"Error: Please select at least one time window\")\n",
                "            return\n",
                "        \n",
                "        # Display training parameters\n",
                "        print(\"\\nStarting Training with the following parameters:\")\n",
                "        print(f\"Flare Classes: {', '.join(flare_classes)}\")\n",
                "        print(f\"Time Windows: {', '.join(time_windows)}\")\n",
                "        print(f\"Version: {'Auto' if version is None else version}\")\n",
                "        print(f\"Description: {'None' if description is None else description}\")\n",
                "        print(f\"Auto-increment: {'Yes' if auto_increment else 'No'}\")\n",
                "        print(\"\\n\" + \"-\"*50 + \"\\n\")\n",
                "        \n",
                "        # Store trained models information\n",
                "        trained_models = []\n",
                "        versions = []\n",
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
                "                    # Run the training function\n",
                "                    model_dir, trained_version = run_training(\n",
                "                        time_window,\n",
                "                        flare_class,\n",
                "                        version=version,\n",
                "                        description=description,\n",
                "                        auto_increment=auto_increment\n",
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
                "        \n",
                "        # Compare models if requested\n",
                "        if compare_checkbox.value and trained_models:\n",
                "            print(\"\\nComparing trained models...\")\n",
                "            try:\n",
                "                from models.model_tracking import compare_models\n",
                "                comparison = compare_models(\n",
                "                    list(set(versions)),  # Unique versions\n",
                "                    list(flare_classes),\n",
                "                    list(time_windows)\n",
                "                )\n",
                "                print(\"\\nModel Comparison:\")\n",
                "                print(comparison)\n",
                "            except Exception as e:\n",
                "                print(f\"Error comparing models: {e}\")\n",
                "        \n",
                "        print(\"\\nTraining completed!\")\n",
                "\n",
                "# Create the run button\n",
                "run_button = widgets.Button(\n",
                "    description='Start Training',\n",
                "    button_style='success',\n",
                "    tooltip='Run training with selected parameters'\n",
                ")\n",
                "\n",
                "# Attach the click handler\n",
                "run_button.on_click(run_training_with_params)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Training Interface\n",
                "\n",
                "Select your training parameters below and click 'Start Training':"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Display the controls\n",
                "display(widgets.VBox([\n",
                "    widgets.HBox([flare_dropdown, time_window_dropdown]),\n",
                "    widgets.HBox([version_text, description_text]),\n",
                "    widgets.HBox([auto_increment_checkbox, compare_checkbox]),\n",
                "    run_button\n",
                "]))\n",
                "\n",
                "# Display the output area\n",
                "display(output)"
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