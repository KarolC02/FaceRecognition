#!/bin/bash
# Wait for training to complete, then evaluate all models

echo "Waiting for training to complete..."
while pgrep -f "training/train.py" > /dev/null; do
    sleep 10
    echo -n "."
done
echo ""
echo "Training complete! Starting evaluation..."

cd /home/cichork/FaceRecognition

# Evaluate ResNet18
echo "Evaluating ResNet18..."
.conda/bin/python scripts/evaluate_test_set.py \
    --model-path data/models/best_model_resnet18.pth \
    --output test_results_resnet18_5epochs.json

# Evaluate FaceNet (trained from scratch)
echo "Evaluating FaceNet (trained from scratch)..."
.conda/bin/python scripts/evaluate_test_set.py \
    --model-path data/models/best_model_facenet.pth \
    --output test_results_facenet_5epochs.json

# Evaluate FaceNet (fine-tuned)
echo "Evaluating FaceNet (fine-tuned)..."
.conda/bin/python scripts/evaluate_test_set.py \
    --model-path data/models/best_model_facenet.pth \
    --output test_results_facenet_finetuned_5epochs.json

echo "All evaluations complete!"

