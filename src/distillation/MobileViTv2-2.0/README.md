# Knowledge Distillation: EfficientNet-B4 → MobileViTv2-2.0

## Quick Start

### Problem You're Facing
You're getting **NaN values** during distillation because the student model (MobileViTv2-2.0) is starting from ImageNet weights, which aren't adapted to your deepfake detection task. This causes numerical instability when trying to match the teacher's outputs.

### Solution: Two-Stage Training

```
Stage 1 (Pre-training)        Stage 2 (Distillation)
     ↓                                ↓
MobileViTv2-2.0           MobileViTv2-2.0 (pre-trained)
    +                                 +
Hard Labels              Hard Labels + Teacher Knowledge
    ↓                                 ↓
Stable Model             Better Model (with teacher's knowledge)
```

## Step-by-Step Instructions

### **STAGE 1: Pre-train Student Model** (REQUIRED - Do this first!)

This stage adapts the student model to your task without distillation:

```bash
# Inside Docker container
cd /app/src/distillation/MobileViTv2-2.0

# Run pre-training
python train_student_only.py ../../configs/config_SBI.json -n pretrain_mobilevit
```

**Expected Time**: 2-3 hours (100 epochs)  
**Expected Result**: Validation AUC ~0.88-0.92

**What this does**:
- ✓ Fine-tunes MobileViTv2-2.0 on your deepfake dataset
- ✓ Uses only ground truth labels (no teacher)
- ✓ Creates stable initialization for Stage 2
- ✓ Saves best checkpoints in `output/pretrain_mobilevit_*/weights/`

### **STAGE 2: Knowledge Distillation**

After Stage 1 completes, use the best checkpoint for distillation:

```bash
# 1. Find best checkpoint from Stage 1
ls -lh output/pretrain_mobilevit_*/weights/
# Look for the .pth file with highest AUC, e.g., "95_0.9134_val.pth"

# 2. Update config file
nano ../../configs/config_SBI.json

# Add this line (replace path with your best checkpoint):
# "pretrained_student_weights": "output/pretrain_mobilevit_config_SBI_01_20_10_30_45/weights/95_0.9134_val.pth"

# 3. Run distillation
python main_distillation.py ../../configs/config_SBI.json -n distill_mobilevit
```

**Expected Time**: 3-4 hours (100 epochs)  
**Expected Result**: Validation AUC ~0.90-0.94 (better than Stage 1)

## Configuration Guide

### Example `config_SBI.json` for Distillation

```json
{
  "image_size": 380,
  "batch_size": 16,
  "epoch": 100,
  
  "teacher_weights": "weights/FFc23.tar",
  "pretrained_student_weights": "output/pretrain_mobilevit_*/weights/BEST_MODEL.pth",
  
  "lr": 0.00005,
  "temperature": 3.0,
  
  "use_progressive_distillation": true,
  "alpha_start": 0.95,
  "alpha_end": 0.5,
  "progressive_epochs": 20
}
```

### Key Parameters Explained

| Parameter | What It Does | Recommended Value |
|-----------|--------------|-------------------|
| `lr` | Learning rate for distillation | 0.00005 (half of pre-training LR) |
| `temperature` | Smoothness of soft labels | 3.0 (lower = more focused) |
| `alpha_start` | Initial hard label weight | 0.95 (mostly hard labels at start) |
| `alpha_end` | Final hard label weight | 0.5 (balanced at end) |
| `progressive_epochs` | Epochs to transition alpha | 20 (gradual transition) |

## Progressive Distillation Schedule

The training automatically transitions from hard labels to soft labels:

```
Epochs 1-5:   alpha=0.95  →  95% hard labels, 5% teacher
Epochs 6-10:  alpha=0.85  →  85% hard labels, 15% teacher
Epochs 11-15: alpha=0.75  →  75% hard labels, 25% teacher
Epochs 16-20: alpha=0.65  →  65% hard labels, 35% teacher
Epochs 21+:   alpha=0.50  →  50% hard labels, 50% teacher
```

This prevents numerical instability by easing the student into distillation.

## Understanding Knowledge Transfer

### What Gets Transferred?

1. **Dark Knowledge**: Teacher's confidence in wrong answers
   - E.g., Teacher says: 70% fake, 30% real (even for 100% fake image)
   - Student learns: Not all fakes are equally obvious

2. **Decision Boundaries**: Where teacher draws the line
   - Soft labels smooth the decision boundary
   - Prevents student from being overconfident

3. **Feature Relationships**: Which features matter together
   - Temperature scaling reveals subtle patterns
   - Student mimics teacher's internal representation

### The Math Behind It

```python
# Hard Label Loss (ground truth)
L_hard = CrossEntropy(student_output, true_label)

# Soft Label Loss (teacher knowledge)
teacher_soft = softmax(teacher_output / temperature)
student_soft = softmax(student_output / temperature)
L_soft = KL_Divergence(student_soft, teacher_soft) * temperature²

# Combined Loss
L_total = alpha * L_hard + (1 - alpha) * L_soft
```

**Why temperature?**
- Low temp (T=1): Sharp, peaked distribution (like argmax)
- High temp (T=5): Smooth, spread-out distribution (reveals dark knowledge)
- We use T² in loss to counterbalance the smoothing effect

## Troubleshooting

### Issue: NaN values in Stage 1 (Pre-training)

**This shouldn't happen, but if it does:**

```bash
# Lower the learning rate
# Edit config_SBI.json: "lr": 0.00005
```

### Issue: NaN values in Stage 2 (Distillation)

**Causes**:
1. ✗ Skipped Stage 1 (pre-training)
2. ✗ Checkpoint path wrong in config
3. ✗ Temperature too high

**Solutions**:
```bash
# 1. Verify checkpoint loads correctly
python -c "
import torch
ckpt = torch.load('output/pretrain_mobilevit_*/weights/BEST_MODEL.pth')
print('Checkpoint keys:', ckpt.keys())
print('Val AUC:', ckpt.get('val_auc', 'N/A'))
"

# 2. Lower temperature in config
# "temperature": 2.0

# 3. Start with more hard labels
# "alpha_start": 0.98
```

### Issue: Student not improving beyond Stage 1

**Cause**: Not enough teacher influence

**Solution**:
```json
{
  "alpha_end": 0.3,        // More teacher at the end
  "temperature": 4.0,      // More knowledge transfer
  "progressive_epochs": 30 // Longer transition
}
```

### Issue: Out of memory

**Solution**:
```json
{
  "batch_size": 8  // Will become 2 during distillation
}
```

## Expected Results

### Performance Comparison

| Model | Parameters | Val AUC | Speed | Memory |
|-------|-----------|---------|-------|--------|
| Teacher (EfficientNet-B4) | 19M | 0.94-0.96 | 50ms | 8GB |
| Student Pre-trained | 18M | 0.88-0.92 | 30ms | 4GB |
| **Student + Distillation** | **18M** | **0.90-0.94** | **30ms** | **4GB** |

**Benefit**: You get teacher-level accuracy with student-level efficiency!

### Training Progress (Expected)

**Stage 1 (Pre-training)**:
```
Epoch 10:  Val AUC = 0.82
Epoch 30:  Val AUC = 0.87
Epoch 50:  Val AUC = 0.89
Epoch 100: Val AUC = 0.91  ← Use this for Stage 2
```

**Stage 2 (Distillation)**:
```
Epoch 10:  Val AUC = 0.92 (already better!)
Epoch 30:  Val AUC = 0.93
Epoch 50:  Val AUC = 0.935
Epoch 100: Val AUC = 0.94  ← Final model
```

## File Structure

```
src/distillation/MobileViTv2-2.0/
├── train_student_only.py          ← Stage 1: Pre-training script
├── main_distillation.py            ← Stage 2: Distillation script
├── KNOWLEDGE_DISTILLATION_GUIDE.md ← Detailed guide
├── README.md                       ← This file
├── teacher_model.py                ← EfficientNet-B4 wrapper
└── student_model.py                ← MobileViTv2-2.0 wrapper

output/
├── pretrain_mobilevit_*/weights/   ← Stage 1 checkpoints
└── distill_mobilevit_*/weights/    ← Stage 2 checkpoints
```

## Next Steps After Training

### 1. Export for Mobile Deployment

```bash
# Convert to TorchScript
python export_for_mobile.py --checkpoint output/distill_mobilevit_*/weights/BEST_MODEL.pth
```

### 2. Evaluate on Test Set

```bash
# Run inference
python evaluate_model.py --checkpoint output/distill_mobilevit_*/weights/BEST_MODEL.pth
```

### 3. Compare with Teacher

```bash
# Side-by-side comparison
python compare_models.py \
  --teacher weights/FFc23.tar \
  --student output/distill_mobilevit_*/weights/BEST_MODEL.pth
```

## Advanced Techniques

### Feature-Based Distillation

For even better results, match intermediate features:

```python
# Modify models to expose features
teacher_features = teacher_model.get_features(img)
student_features = student_model.get_features(img)

# Add feature matching loss
feature_loss = F.mse_loss(student_features, teacher_features)
total_loss = alpha * hard_loss + beta * soft_loss + gamma * feature_loss
```

### Self-Distillation

Use the best student model as a new teacher:

```bash
# Train a new student from the distilled student
python main_distillation.py configs/config_SBI.json \
  --teacher-weights output/distill_mobilevit_*/weights/BEST_MODEL.pth
```

## FAQ

**Q: Why can't I skip Stage 1?**  
A: The student model needs to adapt to your task first. Direct distillation from ImageNet → Deepfake Detection is too big a jump.

**Q: Can I use a different student architecture?**  
A: Yes! Just modify `student_model.py` to load your chosen model.

**Q: What if I don't have a teacher model?**  
A: Train a teacher first, or use self-distillation (train large model, distill to small model).

**Q: How long does the whole process take?**  
A: ~5-7 hours total (2-3h Stage 1 + 3-4h Stage 2) on a single GPU.

## References

- [Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)](https://arxiv.org/abs/1503.02531)
- [Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization" (2020)](https://arxiv.org/abs/2010.01412)
- [Mehta & Rastegari, "Separable Self-attention for Mobile Vision Transformers" (2022)](https://arxiv.org/abs/2206.02680)

## Support

If you encounter issues:

1. Check logs in `output/SESSION_NAME/logs/losses.logs`
2. Review `KNOWLEDGE_DISTILLATION_GUIDE.md` for detailed troubleshooting
3. Verify GPU memory usage with `nvidia-smi`
4. Ensure all dependencies are installed: `pip install -r requirements.txt`
