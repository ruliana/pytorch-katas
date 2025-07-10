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
- **Characters and their detailed personas:**
  - **Master Pai-Torch**: Ancient grandmaster, speaks in cryptic koans about gradients and loss functions. Appears mysteriously when stuck on problems. Claims to have invented first neural network using bamboo.
  - **Master Ao-Tougrad**: Mysterious keeper of backpropagation arts, rarely speaks directly but leaves helpful hints. Has unsettling habit of finishing code comments before you write them.
  - **Cook Oh-Pai-Timizer**: Head cook who relates every cooking technique to optimization algorithms. Teaches momentum while stirring soup, explains Adam optimizer while making bread.
  - **He-Ao-World**: Temple janitor (hidden master trope), introduces real-world data problems through well-intentioned "accidents." Always apologetic but timing is oddly convenient.
  - **Suki**: Sacred temple cat, behaviors serve as training data. Rumored to understand tensor operations better than most humans, communicates through cryptic purrs and meows.

**Character Personality Integration Rules:**
- Each character's introduction should feel natural to their established persona
- Problems should emerge organically from their domain expertise
- Learning objectives should align with each character's teaching style
- When multiple characters appear, they should interact authentically
- Avoid forcing characters into situations that don't match their personality

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

## 🏮 The Ancient Scroll Unfurls 🏮

[CREATIVE TITLE IN ALL CAPS]
Dan Level: [X] ([Dan Title]) | Time: [X] minutes | Sacred Arts: [Concept1], [Concept2], [Concept3]
```
**Note**: Replace `dan_X/filename_unrevised.ipynb` with the actual file path for the kata being created.

### 2. Problem Description - DYNAMIC STORY GENERATION
- Start with "## 📜 THE MASTER'S CHALLENGE"
- **USE ORACLE TABLES** to generate unique story combinations (see Story Generation System below)
- Clear learning objectives with checkboxes using "🎯 THE SACRED OBJECTIVES"

## 🎲 STORY GENERATION SYSTEM (MANDATORY)

**STEP 1: Character Selection**
This story should use n = !`random 1 3` characters, use first n numbers below to choose them (ignore repeated numbers)

Characters:!`random 1 3`, !`random 1 3`, !`random 1 3`

**STEP 2: Character Oracle Table**

1. **Master Pai-Torch** - Ancient grandmaster, cryptic koans about gradients and loss
2. **Cook Oh-Pai-Timizer** - Head cook, relates cooking techniques to optimization
3. **Master Ao-Tougrad** - Mysterious backpropagation keeper, leaves helpful hints
4. **He-Ao-World** - Humble janitor (hidden master), introduces real-world data problems
5. **Suki** - Sacred temple cat, behaviors serve as training data

**STEP 3: Plot Oracle Table**

Use the !`random 1 10` plot below:

1. **The Hidden Technique** - Ancient scroll reveals forgotten method
2. **The Broken Weapon** - Essential tool malfunctions, need new approach
3. **The Rival School** - Competing methods, must prove superiority
4. **The Mysterious Visitor** - Stranger brings new challenge/knowledge
5. **The Poisoned Well** - Data corruption threatens the temple
6. **The Stolen Artifact** - Important model/data goes missing
7. **The Final Test** - Master announces ultimate trial
8. **The Forbidden Knowledge** - Dangerous technique must be mastered safely
9. **The Tournament** - Competition forces innovation
10. **The Natural Disaster** - External crisis requires immediate solution

**STEP 4: Story Synthesis**
Combine the selected characters + plot + PyTorch concept into a cohesive narrative.

**Character Interaction Guidelines:**
- **1 Character**: Deep focus on their personality and teaching style
- **2 Characters**: Create natural interaction/dialogue between them
- **3 Characters**: One leads, others provide supporting perspectives
- Maintain each character's authentic voice and domain expertise
- Connect the plot naturally to the PyTorch learning objectives

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
6. **Use proper Jupyter notebook JSON format** (CRITICAL - see format requirements below)

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
