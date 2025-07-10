---
description: Generate a complete PyTorch kata following the temple's sacred structure
allowed-tools: 
  - Write
  - LS
  - Bash
  - Read
  - Edit
  - Glob
---

You are tasked with creating a PyTorch Code Kata for the "Temple of Neural Networks" repository. This kata must follow the exact structure and requirements outlined in the CLAUDE.md file.

## Command Parameters
- **Dan Level**: $ARGUMENTS (first argument - must be 1-5)
- **Optional Topic**: $ARGUMENTS (second argument - optional specific concept/topic)

## Character Guidelines (CRITICAL - MUST FOLLOW)
- All characters are gender-neutral except Suki (female cat)
- Never use "he" or "she" for masters, cook, or janitor - only use their names
- Always address the learner as "you" or "Grasshopper"
- **Characters and their roles:**
  - **Master Pai-Torch**: Ancient grandmaster, speaks in cryptic koans about gradients
  - **Master Ao-Tougrad**: Mysterious keeper of backpropagation arts, leaves helpful hints
  - **Cook Oh-Pai-Timizer**: Head cook who relates cooking to optimization algorithms
  - **He-Ao-World**: Temple janitor (hidden master), introduces real-world data problems through "clumsiness"
  - **Suki**: Sacred temple cat, behaviors serve as training data

## Dan Level Specifications
- **Dan 1 (Temple Sweeper)**: Tensor operations, linear relationships, basic training loops
- **Dan 2 (Temple Guardian)**: Regularization, validation, multiple layers, optimization
- **Dan 3 (Weapon Master)**: CNNs, RNNs, attention mechanisms, transfer learning
- **Dan 4 (Combat Innovator)**: Custom loss functions, multi-task learning, adversarial training
- **Dan 5 (Mystic Arts Master)**: GANs, diffusion models, graph networks, meta-learning

## Required Kata Structure (ALL 7 COMPONENTS MANDATORY)

### 1. Header Section
Create with this exact format (including Google Colab badge):
```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ruliana/pytorch-katas/blob/main/dan_X/filename_unrevised.ipynb)

🏮 The Ancient Scroll Unfurls 🏮

[CREATIVE TITLE IN ALL CAPS]
Dan Level: [X] ([Dan Title]) | Time: [X] minutes | Sacred Arts: [Concept1], [Concept2], [Concept3]
```
**Note**: Replace `dan_X/filename_unrevised.ipynb` with the actual file path for the kata being created.

### 2. Problem Description
- Start with "📜 THE MASTER'S CHALLENGE"
- Engaging narrative involving the characters
- Include He-Ao-World causing a data problem through "clumsiness"
- Master Pai-Torch providing cryptic wisdom
- Clear learning objectives with checkboxes using "🎯 THE SACRED OBJECTIVES"

### 3. Synthetic Dataset Generator
- Function with themed naming (e.g., "generate_cat_feeding_data")
- Include parameters for difficulty scaling
- Add visualization function with temple theming
- Complete working code with docstrings
- No external data dependencies

### 4. Starter Code Template
- PyTorch class with themed name
- Clear TODO comments with hints
- Training function with gradient management
- Proper error handling and progress reporting
- Include mystical/temple-themed variable names

### 5. Success Criteria
- "⚡ THE TRIALS OF MASTERY" section
- Quantitative metrics (loss targets, accuracy thresholds)
- Unit test function called "test_your_wisdom"
- Parameter validation matching expected values
- Convergence behavior requirements

### 6. Progressive Extensions (4 EXTENSIONS REQUIRED)
- "🌸 THE FOUR PATHS OF MASTERY: PROGRESSIVE EXTENSIONS"
- Each extension introduces a character with new challenges
- Progressive difficulty increase (+15%, +25%, +35%, +45%)
- New concepts for each extension
- Clear success criteria for each

### 7. Debugging Challenge
- "🔥 CORRECTING YOUR FORM: A STANCE IMBALANCE" section
- Intentionally flawed code with common mistakes
- Master Pai-Torch providing guidance
- Clear hints for identification and fixing
- Focused on gradient/training issues

## Technical Requirements
- All code must use PyTorch (even for simple problems)
- Include proper imports at the beginning
- Use torch.manual_seed for reproducibility
- Include matplotlib for visualizations
- Focus on building PyTorch muscle memory
- Code should be pedagogically sound

## Output Requirements
1. **Check dan level validity** (1-5)
2. **Create complete Jupyter notebook** with proper cell structure
3. **Place in correct folder**: `dan_[X]/[generated_filename]_unrevised.ipynb`
4. **Include Google Colab badge** at the top pointing to repository `ruliana/pytorch-katas`
5. **Interleave markdown and code cells** for storytelling
6. **Use proper Jupyter notebook JSON format**

## File Naming Convention
- Use descriptive names related to the problem
- Include "_unrevised" suffix
- Example: `dan_1/temple_cat_feeding_predictor_unrevised.ipynb`

## Important Notes
- The kata should be challenging but achievable for the dan level
- Include proper error checking and validation
- All characters should feel authentic to their personalities
- The narrative should be engaging and cohesive
- Code should teach PyTorch fundamentals through practice

## CRITICAL: Uniqueness and Documentation Requirements

### 1. Check for Existing Katas (MANDATORY)
Before creating a new kata, you MUST:
- Read the README.md file to see all existing katas listed for the target dan level
- Analyze the concepts already covered in existing katas from the README descriptions
- Ensure your new kata covers DIFFERENT concepts and doesn't repeat existing lessons
- If no katas are listed for the dan level, proceed with any appropriate concept for that level

### 2. README.md Updates (MANDATORY)
After creating the kata, you MUST:
- Read the current README.md file
- Add an entry for the new kata in the appropriate dan level section
- Include the kata title, brief description, and file link
- Follow the existing formatting style in README.md
- Use proper markdown formatting for the kata entry

### 3. Concept Differentiation Strategy
- For Dan 1: Focus on different aspects of basic PyTorch (tensors, linear layers, loss functions, optimizers, etc.)
- For Dan 2: Explore different regularization techniques, architectures, or optimization methods
- For Dan 3: Cover different specialized architectures (CNNs, RNNs, Transformers, etc.)
- For Dan 4: Explore different advanced techniques (custom losses, multi-task learning, etc.)
- For Dan 5: Cover different generative or advanced modeling approaches

## Final Instructions

**Think hard** about creating a unique and valuable kata that complements the existing ones. Consider the learning journey and what concepts would be most beneficial for practitioners at this dan level.

Now create a complete PyTorch kata following these specifications:

1. **FIRST**: Read README.md to check existing katas for the target dan level to avoid duplication
2. **SECOND**: Validate the dan level (1-5)
3. **THIRD**: Generate the complete Jupyter notebook with all required sections
4. **FOURTH**: Add the new kata to README.md for easy navigation

If a topic is provided, incorporate it naturally into the kata while maintaining the temple theme and character interactions.