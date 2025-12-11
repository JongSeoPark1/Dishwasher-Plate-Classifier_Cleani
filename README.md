Install dependencies

pip install -r requirements.txt

-----------------------------------------------------------------------------------------------------------------

Prepare Dataset Place your images in the data/ directory following the structure below:

data/pretrain/train/cleaned, data/pretrain/train/dirty

data/finetune/train/cleaned, data/finetune/train/dirty

data/test/cleaned, data/test/dirty

-----------------------------------------------------------------------------------------------------------------
⚙️ Usage / Training Pipeline
This project uses a Two-Stage Training Strategy to handle domain shifts between web-crawled data and real-world appliance data.
-----------------------------------------------------------------------------------------------------------------

Stage 1: Pre-training (Base Learning)

Trains the models from scratch (or ImageNet weights) using a large dataset.

python train_base.py

Output: Models are saved in saved_models/.

-----------------------------------------------------------------------------------------------------------------

Stage 2: Fine-tuning (Domain Adaptation)

Refines the models using real-world data collected from the dishwasher environment with a lower learning rate (1e-5).

python train_finetune.py

Output: Final models are saved in finetuned_models/.

-----------------------------------------------------------------------------------------------------------------
Stage 3: Inference & Testing

Evaluates the ensemble model on the test dataset.

python inference.py

Output: Prints prediction results, confidence scores, and final accuracy.

-----------------------------------------------------------------------------------------------------------------
📊 Model Details
Model	                                     
MobileNetV3-Large: Summurized information utilization.

EfficientNet-B0: Characteristic of images.

ResNet18:	Robust and stable feature extraction.

Ensemble Averages Softmax probabilities from all 3 models.

🏆 Results
Test Accuracy: 94.4%

Inference Speed: Real-time processing capable on CPU

Robustness: Successfully classifies plates under various conditions inside the dishwasher.
