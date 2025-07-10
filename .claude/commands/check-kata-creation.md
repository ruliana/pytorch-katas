---
description: Thoroughly validate the most recent kata creation against all requirements
allowed-tools: 
  - Read
  - NotebookRead
  - NotebookWrite
  - Bash
  - LS
  - Glob
---

**Think hard** about conducting a comprehensive validation of the most recently created PyTorch kata. This command performs a thorough quality assurance check against all requirements from CLAUDE.md and the sacred temple standards.

## Validation Mission

You must conduct a complete inspection of the most recent kata creation to ensure it meets all sacred temple requirements. This is a critical quality control step before the kata can be considered ready for disciples.

## Step-by-Step Validation Process

### 1. **FIRST: Identify the Most Recent Kata**
- Use git log or file timestamps to find the most recently created kata file
- Look for files with `_unrevised.ipynb` suffix
- Confirm the file exists and is a valid Jupyter notebook in proper JSON format
- Check that it contains properly formatted cells, not raw markdown text

### 2. **SECOND: Comprehensive Structure Validation**

#### Header Section Requirements ✅/❌
- [ ] **Google Colab Badge**: Correct format with `ruliana/pytorch-katas` repository
- [ ] **Temple Formatting**: "🏮 The Ancient Scroll Unfurls 🏮"
- [ ] **Title**: Creative, all caps, descriptive
- [ ] **Dan Level**: Correct format "Dan Level: [X] ([Dan Title])"
- [ ] **Time Estimate**: Realistic completion time
- [ ] **Sacred Arts**: 3-4 relevant PyTorch concepts listed

#### Problem Description Requirements ✅/❌
- [ ] **Opening**: Starts with "📜 THE MASTER'S CHALLENGE"
- [ ] **Character Integration**: All characters used appropriately and authentically
- [ ] **Story Uniqueness**: Uses different character combinations than previous katas
- [ ] **Plot Variety**: Uses different plot framework than recent katas
- [ ] **Character Authenticity**: Characters behave according to their established personas
- [ ] **Oracle System Usage**: Evidence of using !`random` commands for story generation
- [ ] **Narrative Flow**: Engaging story that sets up the problem naturally
- [ ] **Sacred Objectives**: Clear checkboxes with "🎯 THE SACRED OBJECTIVES"

#### Synthetic Dataset Generator Requirements ✅/❌
- [ ] **Function Name**: Themed naming related to the problem
- [ ] **Parameters**: Includes difficulty scaling parameters
- [ ] **Docstring**: Complete with Args and Returns
- [ ] **Implementation**: Working code with proper tensor operations
- [ ] **Visualization**: Companion visualization function
- [ ] **No External Dependencies**: Uses only synthetic data generation

#### Starter Code Template Requirements ✅/❌
- [ ] **PyTorch Class**: Properly structured nn.Module
- [ ] **TODO Comments**: Clear instructions with hints
- [ ] **Training Function**: Includes gradient management TODOs
- [ ] **Mystical Naming**: Temple-themed variable and function names
- [ ] **Progressive Structure**: Logical learning progression

#### Success Criteria Requirements ✅/❌
- [ ] **Section Title**: "⚡ THE TRIALS OF MASTERY"
- [ ] **Quantitative Metrics**: Loss targets, accuracy thresholds
- [ ] **Test Function**: Called "test_your_wisdom" or similar
- [ ] **Parameter Validation**: Checks learned weights/biases
- [ ] **Convergence Requirements**: Clear success indicators

#### Progressive Extensions Requirements ✅/❌
- [ ] **Section Title**: "🌸 THE FOUR PATHS OF MASTERY: PROGRESSIVE EXTENSIONS"
- [ ] **Four Extensions**: Exactly 4 progressive extensions
- [ ] **Character Integration**: Each extension introduces a character
- [ ] **Difficulty Progression**: +15%, +25%, +35%, +45% increases
- [ ] **New Concepts**: Each extension teaches additional PyTorch concepts
- [ ] **Clear Success Criteria**: Each extension has defined success metrics

#### Debugging Challenge Requirements ✅/❌
- [ ] **Section Title**: "🔥 CORRECTING YOUR FORM: A STANCE IMBALANCE"
- [ ] **Flawed Code**: Intentionally broken training function
- [ ] **Master Pai-Torch Guidance**: Wisdom about the error
- [ ] **Common Mistake**: Focuses on typical beginner errors
- [ ] **Hints**: Clear guidance for identification and fixing

### 3. **THIRD: Character Usage Validation**

#### Character Guidelines Compliance ✅/❌
- [ ] **Gender Neutrality**: All characters except Suki are gender-neutral
- [ ] **Proper Addressing**: No "he/she" for masters, cook, or janitor
- [ ] **Grasshopper**: Learner always addressed as "you" or "Grasshopper"
- [ ] **Character Authenticity**: Each character feels true to their personality
- [ ] **Narrative Integration**: Characters naturally advance the story

#### Character Role Verification ✅/❌
- [ ] **Master Pai-Torch**: Provides cryptic wisdom about gradients/concepts
- [ ] **Master Ao-Tougrad**: Hints about backpropagation (if relevant)
- [ ] **Cook Oh-Pai-Timizer**: Relates cooking to optimization/algorithms
- [ ] **He-Ao-World**: Introduces real-world data problems through "accidents"
- [ ] **Suki**: If present, behaviors serve as training data

### 4. **FOURTH: Technical Accuracy Validation**

#### Import Requirements ✅/❌
- [ ] **Core PyTorch**: `torch`, `torch.nn as nn`, `torch.optim as optim`
- [ ] **Visualization**: `matplotlib.pyplot as plt`
- [ ] **Utilities**: `from typing import Tuple`, `import numpy as np`
- [ ] **Reproducibility**: `torch.manual_seed(42)` or similar
- [ ] **Themed Section**: "🕯️ THE SACRED IMPORTS" or similar

#### PyTorch Best Practices ✅/❌
- [ ] **Proper nn.Module**: Correct class structure with `__init__` and `forward`
- [ ] **Gradient Management**: TODOs for `optimizer.zero_grad()`
- [ ] **Loss Functions**: Appropriate loss function selection
- [ ] **Optimizer Usage**: Proper optimizer initialization and stepping
- [ ] **Tensor Operations**: Correct tensor manipulation throughout

#### Educational Value ✅/❌
- [ ] **Dan Level Appropriate**: Concepts match the specified dan level
- [ ] **PyTorch Focus**: Uses neural networks even for simple problems
- [ ] **Muscle Memory**: Builds PyTorch fluency through practice
- [ ] **Clear Learning Path**: Logical progression from basic to advanced
- [ ] **Debugging Skills**: Includes common pitfalls and solutions

### 5. **FIFTH: Documentation & Organization**

#### File Management ✅/❌
- [ ] **Correct Location**: Placed in proper `dan_X/` folder
- [ ] **Naming Convention**: Descriptive name with `_unrevised.ipynb` suffix
- [ ] **Jupyter Format**: Valid notebook with proper JSON structure (not raw markdown)
- [ ] **Cell Types**: Appropriate mix of markdown and code cells with proper formatting
- [ ] **Headers in Cells**: All headers (like "🏮 The Ancient Scroll Unfurls 🏮") are in markdown cells

#### README Integration ✅/❌
- [ ] **Entry Added**: New kata listed in README.md
- [ ] **Proper Section**: Added to correct dan level section
- [ ] **Description**: Includes title, brief description, and concepts
- [ ] **Link Format**: Correct markdown link to the notebook file
- [ ] **Formatting**: Follows existing README style

### 6. **SIXTH: Uniqueness & Quality**

#### Concept Differentiation ✅/❌
- [ ] **Unique Problem**: Different from existing katas in same dan level
- [ ] **Fresh Perspective**: Novel approach to teaching PyTorch concepts
- [ ] **Complementary Learning**: Adds value to existing kata collection
- [ ] **Appropriate Difficulty**: Matches dan level expectations

## Final Validation Report

After completing all checks, provide a comprehensive report with:

1. **Overall Score**: X/Y requirements met
2. **Critical Issues**: Any missing mandatory components
3. **Recommendations**: Specific improvements needed
4. **Strengths**: What the kata does particularly well
5. **Ready for Review**: Yes/No assessment with reasoning

## Quality Standards

- **Excellent (95-100%)**: Ready for immediate use
- **Good (85-94%)**: Minor tweaks needed
- **Needs Work (70-84%)**: Significant improvements required
- **Major Issues (<70%)**: Substantial rework needed

Remember: This validation ensures that every kata maintains the sacred temple standards and provides disciples with the highest quality learning experience. Be thorough, be honest, and uphold the temple's honor!

Now begin the comprehensive validation of the most recent kata creation.
