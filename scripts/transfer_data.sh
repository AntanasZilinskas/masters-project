#!/bin/bash
# Script to transfer data and code to Imperial HPC for EVEREST model training
# Run this script from your laptop after connecting to the Imperial VPN

# Variables - configure these if needed
IMPERIAL_USER="az2221"  # Your Imperial username
LOCAL_DATA_PATH="./data"  # Path to your data folder
LOCAL_NATURE_DATA="./Nature_data"  # Path to your Nature_data folder

# Check if Imperial VPN is connected
echo "Checking connection to Imperial..."
if ! ping -c 1 login.hpc.imperial.ac.uk &> /dev/null; then
    echo "Error: Cannot connect to Imperial HPC. Please connect to Imperial VPN first."
    exit 1
fi

# First, ensure the destination directories exist
ssh ${IMPERIAL_USER}@login.hpc.imperial.ac.uk "mkdir -p ~/projects/everest/{data,scripts,src/models,src/utils,Nature_data}"

# Transfer Nature_data (CSV files)
echo "Transferring Nature_data to Imperial HPC..."
echo "From: ${LOCAL_NATURE_DATA}"
echo "To: ${IMPERIAL_USER}@login.hpc.imperial.ac.uk:~/projects/everest/Nature_data"
rsync -avz --progress ${LOCAL_NATURE_DATA}/ ${IMPERIAL_USER}@login.hpc.imperial.ac.uk:~/projects/everest/Nature_data/

# Transfer specific required files regardless of code transfer choice
echo "Transferring essential model files..."
rsync -avz --progress ./models/train_all_everest.py ${IMPERIAL_USER}@login.hpc.imperial.ac.uk:~/projects/everest/src/models/
rsync -avz --progress ./models/train_complete_everest.py ${IMPERIAL_USER}@login.hpc.imperial.ac.uk:~/projects/everest/src/models/
rsync -avz --progress ./models/complete_everest.py ${IMPERIAL_USER}@login.hpc.imperial.ac.uk:~/projects/everest/src/models/
rsync -avz --progress ./utils/utils.py ${IMPERIAL_USER}@login.hpc.imperial.ac.uk:~/projects/everest/src/utils/

# If you want to transfer your complete code instead of cloning it (for private repo)
echo "Do you want to transfer your complete code instead of cloning it later? (y/n)"
read -p "> " TRANSFER_CODE
if [[ $TRANSFER_CODE == "y" ]]; then
    echo "Transferring code to Imperial HPC..."
    echo "Transferring models directory..."
    rsync -avz --progress ./models/ ${IMPERIAL_USER}@login.hpc.imperial.ac.uk:~/projects/everest/src/models/
    
    echo "Transferring utils directory..."
    rsync -avz --progress ./utils/ ${IMPERIAL_USER}@login.hpc.imperial.ac.uk:~/projects/everest/src/utils/
    
    echo "Transferring requirements file..."
    rsync -avz --progress ./requirements_full.txt ${IMPERIAL_USER}@login.hpc.imperial.ac.uk:~/projects/everest/src/
fi

echo "Transferring data to Imperial HPC..."
echo "From: ${LOCAL_DATA_PATH}"
echo "To: ${IMPERIAL_USER}@login.hpc.imperial.ac.uk:~/projects/everest/data"
rsync -avz --progress ${LOCAL_DATA_PATH}/ ${IMPERIAL_USER}@login.hpc.imperial.ac.uk:~/projects/everest/data/

echo "Transferring scripts to Imperial HPC..."
rsync -avz --progress ./scripts/everest_train.pbs ./scripts/hpc_setup.sh ./scripts/train_all_everest_models.sh ${IMPERIAL_USER}@login.hpc.imperial.ac.uk:~/projects/everest/scripts/

# Make the scripts executable
ssh ${IMPERIAL_USER}@login.hpc.imperial.ac.uk "chmod +x ~/projects/everest/scripts/*.sh"

# Create __init__.py files to ensure modules can be imported
ssh ${IMPERIAL_USER}@login.hpc.imperial.ac.uk "touch ~/projects/everest/src/__init__.py ~/projects/everest/src/models/__init__.py ~/projects/everest/src/utils/__init__.py"

echo "Data transfer complete!"
echo ""
echo "Next steps:"
echo "1. Log into the HPC cluster: ssh ${IMPERIAL_USER}@login.hpc.imperial.ac.uk"
echo "2. Run the setup script: bash ~/projects/everest/scripts/hpc_setup.sh"
echo "3a. To train a single model: cd ~/projects/everest/scripts && qsub -v FLARE_CLASS=M5,TIME_WINDOW=24 everest_train.pbs"
echo "3b. To train all models: cd ~/projects/everest/scripts && ./train_all_everest_models.sh" 