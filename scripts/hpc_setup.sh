#!/bin/bash
# This script sets up the environment for EVEREST model training on Imperial's HPC

# Create project directories with the correct structure for your project
mkdir -p $HOME/projects/everest/{src,data,results,logs,scripts,models/trained_models,weights,Nature_data}

# Clone private repository into src directory (if not already done)
if [ ! -d "$HOME/projects/everest/src/.git" ] && [ ! -f "$HOME/projects/everest/src/models/train_all_everest.py" ]; then
  echo "Repository not found. Choosing setup method..."
  
  echo "Select method to set up your code:"
  echo "1) HTTPS with GitHub username/password"
  echo "2) SSH (if you've already set up SSH keys)"
  echo "3) Use files already transferred (recommended if you ran transfer_data.sh)"
  read -p "Enter your choice (1-3): " CLONE_METHOD
  
  case $CLONE_METHOD in
    1)
      read -p "Enter your GitHub username: " GITHUB_USER
      read -sp "Enter your GitHub personal access token: " GITHUB_TOKEN
      echo ""
      git clone https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/masters-project.git $HOME/projects/everest/src
      ;;
    2)
      read -p "Enter your GitHub username: " GITHUB_USER
      git clone git@github.com:${GITHUB_USER}/masters-project.git $HOME/projects/everest/src
      ;;
    3)
      echo "Using files already transferred. Ensuring proper directory structure..."
      # Create __init__.py files if they don't exist
      touch $HOME/projects/everest/src/__init__.py
      touch $HOME/projects/everest/src/models/__init__.py
      touch $HOME/projects/everest/src/utils/__init__.py
      ;;
    *)
      echo "Invalid option. Please run the script again."
      exit 1
      ;;
  esac
else
  if [ -d "$HOME/projects/everest/src/.git" ]; then
    echo "Repository already exists, updating..."
    cd $HOME/projects/everest/src
    git pull
  else
    echo "Using files already transferred."
  fi
fi

# Verify essential files exist
ESSENTIAL_FILES=(
  "$HOME/projects/everest/src/models/train_all_everest.py"
  "$HOME/projects/everest/src/utils/utils.py"
)

MISSING_FILES=0
for file in "${ESSENTIAL_FILES[@]}"; do
  if [ ! -f "$file" ]; then
    echo "WARNING: Essential file not found: $file"
    MISSING_FILES=1
  fi
done

if [ $MISSING_FILES -eq 1 ]; then
  echo "Some essential files are missing. Please run the transfer_data.sh script from your local machine first."
  read -p "Do you want to continue anyway? (y/n): " CONTINUE
  if [[ $CONTINUE != "y" ]]; then
    echo "Aborting setup. Please run transfer_data.sh first."
    exit 1
  fi
fi

# Copy PBS job script and training scripts
cp $HOME/projects/everest/src/scripts/everest_train.pbs $HOME/projects/everest/scripts/ 2>/dev/null || echo "Warning: Could not copy everest_train.pbs"
cp $HOME/projects/everest/src/scripts/train_all_everest_models.sh $HOME/projects/everest/scripts/ 2>/dev/null || echo "Warning: Could not copy train_all_everest_models.sh"

# Make scripts executable
chmod +x $HOME/projects/everest/scripts/*.sh 2>/dev/null || echo "Warning: Could not make scripts executable"

# Set up Python environment
module load miniforge/3
miniforge-setup
eval "$(~/miniforge3/bin/conda shell.bash hook)"

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "everest_env"; then
  echo "Creating conda environment..."
  conda create -n everest_env -c conda-forge python=3.11 -y
  conda activate everest_env
  
  # Install TensorFlow with compatible CUDA version
  echo "Installing TensorFlow with CUDA..."
  conda install -c conda-forge tensorflow=2.13.0 cudatoolkit=11.8 cudnn=8.7 -y
  
  # Verify TensorFlow installation
  echo "Verifying TensorFlow installation..."
  python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" || echo "WARNING: TensorFlow installation failed!"
  
  if [ -f "$HOME/projects/everest/src/requirements_full.txt" ]; then
    echo "Installing requirements from requirements_full.txt..."
    pip install -r $HOME/projects/everest/src/requirements_full.txt
  else
    echo "Warning: requirements_full.txt not found. Installing essential packages..."
    pip install pandas numpy matplotlib scikit-learn tensorflow-addons
  fi
  
  conda config --set auto_activate_base false
else
  echo "Conda environment already exists"
  conda activate everest_env
  
  # Check if TensorFlow is installed
  if ! python -c "import tensorflow" &> /dev/null; then
    echo "TensorFlow not found in everest_env. Installing..."
    conda install -c conda-forge tensorflow=2.13.0 cudatoolkit=11.8 cudnn=8.7 -y
    pip install tensorflow-addons
  fi
fi

# Create a .bashrc addition to set PYTHONPATH when logging in
cat > $HOME/.everest_pythonpath << EOF
# EVEREST project Python path settings
export PYTHONPATH=\$HOME/projects/everest/src:\$PYTHONPATH
EOF

# Add to .bashrc if not already there
if ! grep -q "EVEREST project Python path" $HOME/.bashrc; then
  echo "" >> $HOME/.bashrc
  echo "# Source EVEREST Python path settings" >> $HOME/.bashrc
  echo "source \$HOME/.everest_pythonpath" >> $HOME/.bashrc
fi

# Set PYTHONPATH for current session
export PYTHONPATH=$HOME/projects/everest/src:$PYTHONPATH

echo ""
echo "Setup complete! You can now:"
echo "1. Train a single model:"
echo "   cd $HOME/projects/everest/scripts"
echo "   qsub -v FLARE_CLASS=M5,TIME_WINDOW=24 everest_train.pbs"
echo ""
echo "2. Or train all models (9 configurations):"
echo "   cd $HOME/projects/everest/scripts"
echo "   ./train_all_everest_models.sh"
echo ""
echo "PYTHONPATH has been set to include $HOME/projects/everest/src"
echo "This will be automatically added to future login sessions." 