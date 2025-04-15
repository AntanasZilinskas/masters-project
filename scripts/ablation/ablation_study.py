#!/usr/bin/env python
'''
Ablation Study for SolarKnowledge Model
This script systematically tests different model configurations for the 24h M-class prediction task
and reports their TSS values for comparison.

Author: Antanas Zilinskas
'''

import sys
import os
import time
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import confusion_matrix

# Add parent directory to path to import from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import model and utilities
try:
    from models.SolarKnowledge_model import SolarKnowledge
    from utils import get_training_data, get_testing_data, data_transform, log
    print("Successfully imported model and utilities")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

# Configure TensorFlow to use Memory Growth if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Memory growth enabled for {len(physical_devices)} GPU devices")
    except Exception as e:
        print(f"Error configuring GPU: {e}")

# Use float32 for all operations
tf.keras.mixed_precision.set_global_policy('float32')

# Ablation study configurations
CONFIGURATIONS = [
    {
        'name': 'Full model',
        'description': '(Conv1D + BN) + LSTM + 4 TEBs + heavy dropout',
        'use_conv': True,
        'use_lstm': True,
        'teb_layers': 4,
        'dropout_rate': 0.2,
        'use_class_weighting': True
    },
    {
        'name': 'No LSTM',
        'description': 'only conv + BN, then TEBs',
        'use_conv': True,
        'use_lstm': False,
        'teb_layers': 4,
        'dropout_rate': 0.2,
        'use_class_weighting': True
    },
    {
        'name': 'No conv',
        'description': 'BN then LSTM',
        'use_conv': False, 
        'use_lstm': True,
        'teb_layers': 4,
        'dropout_rate': 0.2,
        'use_class_weighting': True
    },
    {
        'name': 'Reduced TEBs',
        'description': '2 layers instead of 4',
        'use_conv': True,
        'use_lstm': True,
        'teb_layers': 2,
        'dropout_rate': 0.2,
        'use_class_weighting': True
    },
    {
        'name': 'No class weighting',
        'description': 'No class weights for imbalanced data',
        'use_conv': True,
        'use_lstm': True,
        'teb_layers': 4,
        'dropout_rate': 0.2,
        'use_class_weighting': False
    },
    {
        'name': 'Light dropout',
        'description': 'dropout = 0.1 (lighter)',
        'use_conv': True,
        'use_lstm': True,
        'teb_layers': 4,
        'dropout_rate': 0.1,
        'use_class_weighting': True
    }
]

class AblationModel(SolarKnowledge):
    """Extended SolarKnowledge model with ablation options"""
    
    def build_ablation_model(self, input_shape,
                         embed_dim=128,
                         num_heads=4,
                         ff_dim=256,
                         num_transformer_blocks=4,
                         dropout_rate=0.2,
                         num_classes=2,
                         use_conv=True,
                         use_lstm=True,
                         use_class_weighting=True):
        """Build model with ablation options"""
        # Store architecture details
        self.metadata["model_architecture"] = {
            "num_transformer_blocks": num_transformer_blocks,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "dropout_rate": dropout_rate,
            "use_conv": use_conv,
            "use_lstm": use_lstm,
            "use_class_weighting": use_class_weighting
        }
        
        inputs = tf.keras.layers.Input(shape=input_shape)
        self.input_tensor = inputs
        x = inputs
        
        # Conv1D + BatchNormalization feature extraction
        if use_conv:
            x = tf.keras.layers.Conv1D(filters=embed_dim, kernel_size=3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
        
        # LSTM layer
        if use_lstm:
            x = tf.keras.layers.LSTM(embed_dim, return_sequences=True)(x)
        
        # Dense embedding if no conv or LSTM was used
        if not use_conv and not use_lstm:
            x = tf.keras.layers.Dense(embed_dim)(x)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
            
        # Apply dropout after input processing
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        # Transformer Encoder Blocks (TEBs)
        for i in range(num_transformer_blocks):
            x = self.transformer_block(x, embed_dim, num_heads, ff_dim, dropout_rate)
        
        # Global pooling and classification head
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.use_class_weighting = use_class_weighting
        return self.model
    
    def transformer_block(self, inputs, embed_dim, num_heads, ff_dim, dropout_rate):
        """Transformer encoder block implementation"""
        # Multi-head attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim//num_heads)(inputs, inputs)
        attention = tf.keras.layers.Dropout(dropout_rate)(attention)
        # Add & normalize (residual connection)
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
        
        # Feed-forward network
        ffn = tf.keras.layers.Dense(ff_dim, activation='relu')(attention)
        ffn = tf.keras.layers.Dropout(dropout_rate)(ffn)
        ffn = tf.keras.layers.Dense(embed_dim)(ffn)
        ffn = tf.keras.layers.Dropout(dropout_rate)(ffn)
        # Add & normalize (residual connection)
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + ffn)
    
    def compile(self, loss='categorical_crossentropy', metrics=['accuracy'], learning_rate=1e-4):
        """Compile the model with appropriate settings"""
        self.metadata["training"] = {
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "loss": loss,
            "metrics": metrics,
            "use_class_weighting": self.use_class_weighting
        }
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )


def calculate_tss(y_true, y_pred):
    """Calculate True Skill Statistic (TSS) from binary predictions"""
    cm = confusion_matrix(y_true, y_pred)
    # For binary classification: sensitivity = recall for positive class (1)
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    # Specificity = recall for negative class (0)
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    # TSS = sensitivity + specificity - 1
    return sensitivity + specificity - 1


def run_ablation_study(time_window="24", flare_class="M", epochs=50):
    """Run ablation study for all configurations and report results"""
    results = []
    
    # Load data once
    print(f"Loading data for {time_window}h {flare_class}-class prediction...")
    try:
        X_train, y_train = get_training_data(time_window, flare_class)
        y_train_tr = data_transform(y_train)
        X_test, y_test = get_testing_data(time_window, flare_class)
        
        # Ensure data is in float32 format
        X_train = tf.cast(X_train, tf.float32)
        y_train_tr = tf.cast(y_train_tr, tf.float32)
        X_test = tf.cast(X_test, tf.float32)
        
        print(f"Data loaded successfully:")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train_tr.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {np.array(y_test).shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return []
    
    # Calculate class weights for imbalanced data
    class_counts = np.sum(y_train_tr, axis=0)
    n_samples = len(y_train_tr)
    class_weights = {
        0: n_samples / (2 * class_counts[0]) if class_counts[0] > 0 else 1.0,
        1: n_samples / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0
    }
    
    print(f"Class weights: {class_weights}")
    
    # For simplicity in this example, you can use dummy data with the actual shapes
    # from your model if the real data is not available
    if X_train.shape[0] == 0:
        print("Using dummy data for demonstration")
        X_train = np.random.normal(size=(1000, 10, 14)).astype(np.float32)  # Adjust shapes as needed
        y_train_tr = np.zeros((1000, 2)).astype(np.float32)
        y_train_tr[:, 0] = 1  # All negative class for demo
        y_train_tr[np.random.choice(1000, 100), :] = [0, 1]  # 10% positive class
        
        X_test = np.random.normal(size=(200, 10, 14)).astype(np.float32)
        y_test = np.zeros(200)
        y_test[np.random.choice(200, 20)] = 1  # 10% positive class
    
    # Run ablation for each configuration
    for i, config in enumerate(CONFIGURATIONS):
        print(f"\n[{i+1}/{len(CONFIGURATIONS)}] Testing configuration: {config['name']}")
        print(f"Description: {config['description']}")
        
        # Create model with the specific configuration
        model = AblationModel(early_stopping_patience=5, 
                              description=f"Ablation: {config['description']}")
        
        # Get input shape from the training data
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Build the model with ablation options
        try:
            model.build_ablation_model(
                input_shape=input_shape,
                embed_dim=128,
                num_heads=4,
                ff_dim=256,
                num_transformer_blocks=config['teb_layers'],
                dropout_rate=config['dropout_rate'],
                num_classes=2,
                use_conv=config['use_conv'],
                use_lstm=config['use_lstm'],
                use_class_weighting=config['use_class_weighting']
            )
            
            # Compile the model
            model.compile()
            
            # Add callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=5, 
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=3, 
                    verbose=1, 
                    min_lr=1e-6
                )
            ]
        except Exception as e:
            print(f"Error building model: {e}")
            continue
        
        # Training with or without class weights
        try:
            if config['use_class_weighting']:
                print("Training with class weights")
                history = model.model.fit(
                    X_train, y_train_tr,
                    epochs=epochs,
                    verbose=1,
                    batch_size=512,
                    validation_split=0.2,
                    class_weight=class_weights,
                    callbacks=callbacks
                )
            else:
                print("Training without class weights")
                history = model.model.fit(
                    X_train, y_train_tr,
                    epochs=epochs,
                    verbose=1,
                    batch_size=512,
                    validation_split=0.2,
                    callbacks=callbacks
                )
        except Exception as e:
            print(f"Error during training: {e}")
            # You can choose to continue with next configuration or assign a default TSS
            continue
        
        # Evaluate on test set
        try:
            print("Evaluating on test set...")
            y_pred_probs = model.model.predict(X_test)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            
            # If y_test is one-hot encoded, convert to class indices
            if len(np.array(y_test).shape) > 1 and np.array(y_test).shape[1] > 1:
                y_test_classes = np.argmax(np.array(y_test), axis=1)
            else:
                y_test_classes = np.array(y_test)
            
            # Calculate metrics
            tss = calculate_tss(y_test_classes, y_pred_classes)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            continue
        
        # Store results
        result = {
            'config_name': config['name'],
            'description': config['description'],
            'tss': tss,
            'use_conv': config['use_conv'],
            'use_lstm': config['use_lstm'],
            'teb_layers': config['teb_layers'],
            'dropout_rate': config['dropout_rate'],
            'use_class_weighting': config['use_class_weighting'],
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        results.append(result)
        print(f"Configuration: {config['name']}")
        print(f"TSS: {tss:.4f}")
        
    # Save all results
    if results:
        output_dir = "results/ablation"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        output_file = os.path.join(output_dir, f"ablation_results_{time_window}h_{flare_class}_class.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print formatted summary for LaTeX table
        print("\n\n=== ABLATION STUDY SUMMARY ===")
        print("Results for LaTeX table:")
        print(r"\begin{tabular}{l|c}")
        print(r"\toprule")
        print(r"\textbf{Configuration} & \textbf{TSS} \\")
        print(r"\midrule")
        
        for result in results:
            # Format description to match LaTeX format
            if result['config_name'] == 'Full model':
                print(f"Full model: {result['description']} & {result['tss']:.3f}\\\\")
            else:
                print(f"\\quad - {result['description']} & {result['tss']:.3f}\\\\")
        
        print(r"\bottomrule")
        print(r"\end{tabular}")
    else:
        print("No results were obtained from the ablation study")
    
    return results


if __name__ == "__main__":
    print("Starting ablation study...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    # Run ablation study for 24h M-class prediction
    results = run_ablation_study(time_window="24", flare_class="M", epochs=50)
    
    if results:
        # Print the final TSS values for quick reference
        print("\nFinal TSS values:")
        for result in results:
            print(f"{result['config_name']}: {result['tss']:.3f}")
    else:
        print("Ablation study did not produce any results. Check the errors above.") 