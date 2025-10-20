# Knowledge Distillation Guide: EfficientNet-B4 → MobileViTv2-2.0

## Overview
This guide explains how to transfer knowledge from a pre-trained EfficientNet-B4 teacher model to a MobileViTv2-2.0 student model for deepfake detection.

## Problem: Why NaN Occurs

The NaN/Inf issue happens because:

1. **Logit Mismatch**: The student model (pretrained on ImageNet) produces logits in a different range than expected for binary classification
2. **Immediate Distillation**: Trying to match teacher outputs before the student adapts to your task causes numerical instability
3. **SAM Optimizer**: SAM requires 2 forward passes, amplifying any instability

## Solution: Two-Stage Training Approach

### **Stage 1: Pre-train Student Model (REQUIRED)**

First, train the student model on your dataset **WITHOUT** distillation to adapt it to the task.

```bash
# Run student-only training first
python train_student_only.py configs/config_SBI.json -n pretrain_mobilevit
```

This will:
- Fine-tune MobileViTv2-2.0 on your deepfake detection dataset
- Use only hard labels (no teacher guidance)
- Create a stable initialization for distillation
- Save checkpoints in `output/pretrain_mobilevit_*/weights/`

**Expected Output**: Best model with ~85-90% validation AUC

### **Stage 2: Knowledge Distillation**

After pre-training, use the best checkpoint as initialization for distillation:

```bash
# Update config to include pretrained student weights
# Edit configs/config_SBI.json and add:
# "pretrained_student_weights": "output/pretrain_mobilevit_*/weights/best_model.pth"

# Run distillation training
python main_distillation.py configs/config_SBI.json -n distill_mobilevit
```

## Configuration Parameters

### Key Hyperparameters in `config_SBI.json`

```json
{
  "teacher_weights": "weights/FFc23.tar",
  "pretrained_student_weights": "output/pretrain_mobilevit_*/weights/10_0.9234_val.pth",
  
  "lr": 0.00005,  // Lower LR for distillation (half of pre-training LR)
  "temperature": 3.0,  // Controls soft label smoothness
  
  // Progressive Distillation (Recommended)
  "use_progressive_distillation": true,
  "alpha_start": 0.95,  // Start: 95% hard labels, 5% soft labels
  "alpha_end": 0.5,     // End: 50% hard labels, 50% soft labels
  "progressive_epochs": 20,  // Transition over 20 epochs
  
  "epoch": 100,
  "batch_size": 16,  // Will be reduced to 4 for distillation
  "image_size": 380
}
```

### Progressive Distillation Schedule

| Epoch | Alpha | Hard Labels | Soft Labels (Teacher) |
|-------|-------|-------------|----------------------|
| 1-5   | 0.95  | 95%         | 5%                   |
| 6-10  | 0.85  | 85%         | 15%                  |
| 11-15 | 0.75  | 75%         | 25%                  |
| 16-20 | 0.65  | 65%         | 35%                  |
| 21+   | 0.50  | 50%         | 50%                  |

## Understanding the Distillation Process

### What Happens During Training?

1. **Teacher Forward Pass**:
   ```python
   teacher_logits = teacher_model(img)  # EfficientNet-B4 predictions
   soft_teacher_probs = softmax(teacher_logits / temperature)
   ```

2. **Student Forward Pass**:
   ```python
   student_logits = student_model(img)  # MobileViTv2-2.0 predictions
   soft_student_probs = softmax(student_logits / temperature)
   ```

3. **Loss Computation**:
   ```python
   # Hard label loss (ground truth)
   hard_loss = CrossEntropy(student_logits, true_labels)
   
   # Soft label loss (teacher knowledge)
   soft_loss = KL_Divergence(soft_student_probs, soft_teacher_probs) * T²
   
   # Combined loss
   total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
   ```

4. **SAM Optimization**:
   - First step: Perturb weights to find "worst case" point
   - Second step: Optimize from that point (sharpness-aware)

### Why This Works

- **Temperature (T)**: Softens probability distributions, revealing dark knowledge
  - Higher T → smoother distributions, easier to learn
  - Lower T → sharper distributions, closer to hard labels
  
- **Alpha Blending**: Balances task learning vs. teacher mimicking
  - High alpha → focus on getting task right
  - Low alpha → focus on matching teacher's behavior

- **Progressive Schedule**: Gradual transition prevents shock to student model

## Troubleshooting

### Issue: Still Getting NaN

**Cause**: Student model not pre-trained properly

**Solution**:
1. Check Stage 1 training logs - should reach >80% accuracy
2. Verify checkpoint file exists and loads correctly
3. Try longer pre-training (50+ epochs)

### Issue: Low Validation AUC

**Cause**: Too much reliance on soft labels too early

**Solution**:
1. Increase `alpha_start` to 0.98 (more hard labels initially)
2. Extend `progressive_epochs` to 30
3. Lower temperature to 2.0

### Issue: Student Not Improving Beyond Pre-training

**Cause**: Teacher knowledge not being transferred

**Solution**:
1. Decrease `alpha_end` to 0.3 (more soft labels at the end)
2. Increase temperature to 4.0 (more knowledge transfer)
3. Train longer (100+ epochs)

## Expected Results

### Pre-training (Stage 1)
- **Time**: ~2-3 hours (100 epochs)
- **Best Val AUC**: 0.88-0.92
- **Model Size**: ~18M parameters

### Distillation (Stage 2)
- **Time**: ~3-4 hours (100 epochs)
- **Best Val AUC**: 0.90-0.94 (should exceed Stage 1)
- **Model Size**: ~18M parameters (same)
- **Improvement**: 1-3% AUC gain from teacher knowledge

## Model Comparison

| Model | Parameters | Val AUC | Inference Time |
|-------|-----------|---------|----------------|
| Teacher (EfficientNet-B4) | ~19M | 0.94-0.96 | 50ms |
| Student (MobileViTv2-2.0) Pre-trained | ~18M | 0.88-0.92 | 30ms |
| Student + Distillation | ~18M | 0.90-0.94 | 30ms |

## Advanced: Feature-Based Distillation

For even better results, you can match intermediate features (not just outputs):

```python
# Extract intermediate features
teacher_features = teacher_model.get_features(img)
student_features = student_model.get_features(img)

# Feature matching loss
feature_loss = MSE(student_features, teacher_features)

# Total loss
total_loss = alpha * hard_loss + beta * soft_loss + gamma * feature_loss
```

This requires modifying the model classes to expose intermediate activations.

## References

- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- **SAM Optimizer**: Foret et al., "Sharpness-Aware Minimization" (2020)
- **MobileViTv2**: Mehta & Rastegari, "Separable Self-attention for Mobile Vision Transformers" (2022)
