---
description: Generate a complete PyTorch kata following the temple's sacred structure
allowed-tools: 
  - Write
  - NotebookWrite
  - LS
  - Bash(random:*)
  - Read
  - NotebookRead
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
    - Avoid references to gender traits like "beard"
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
**Note**: Replace `dan_X/kata_nn_filename_unrevised.ipynb` with the actual file path for the kata being created.

### 2. Problem Description - CONCISE STORY GENERATION
- Start with "## 📜 THE CHALLENGE"
- **Create exactly 2 paragraphs** based on user's one-liner input (see Story Creation Approach below)
- Clear learning objectives with checkboxes using "🎯 THE SACRED OBJECTIVES"

## 📜 STORY CREATION APPROACH

**User-Driven Story Generation:**
- User provides a one-liner describing the challenge/scenario
- Expand this into exactly **2 paragraphs** for "📜 THE CHALLENGE" section
- Keep it concise and focused on the PyTorch learning objective

**Story Structure Guidelines:**
- **Paragraph 1**: Introduce the scenario and character(s) naturally
- **Paragraph 2**: Present the specific challenge and learning objective
- Any master wisdom should directly relate to the PyTorch concept being taught
- Avoid lengthy atmospheric descriptions - focus on the technical learning goal

**Character Selection:**
- Choose 1-2 characters maximum that naturally fit the scenario
- Each character should contribute directly to understanding the PyTorch concept
- Maintain authentic character voices but keep interactions brief and relevant

**Technical Integration:**
- Connect the story directly to the PyTorch learning objectives
- Ensure character guidance relates to actual coding challenges
- Keep the narrative engaging but subordinate to the educational goal

### 3. Synthetic Dataset Generator
- Function with themed naming (e.g., "generate_cat_feeding_data")
- Include parameters for difficulty scaling
- Add visualization function with temple theming
- Complete working code with docstrings
- No external data dependencies

**Example Template:**

```python
# 📦 ALL IMPORTS AND CONFIGURATION
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

# Global configuration constants
DEFAULT_CHAOS_LEVEL = 0.1
FEEDING_THRESHOLD = 70  # Hunger level at which Suki appears

print("🏮 The Temple of Neural Networks welcomes you, Grasshopper!")
print(f"PyTorch version: {torch.__version__}")
print("🐱 Suki stirs from her afternoon nap, sensing the approach of learning...")
```

```python
# 🐱 THE SACRED DATA GENERATION SCROLL

def generate_cat_feeding_data(n_observations: int = 100, chaos_level: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate observations of Suki's feeding patterns.

    Ancient wisdom suggests: hunger_level = 2.5 * hours_since_last_meal + 20
    When hunger_level > 70, Suki appears at the food bowl.

    Args:
        n_observations: Number of Suki sightings to simulate
        chaos_level: Amount of feline unpredictability (0.0 = perfectly predictable cat, 1.0 = pure chaos)
        seed: Ensures consistent randomness

    Returns:
        Tuple of (hours_since_last_meal, hunger_level) as sacred tensors
    """
    # Suki can go 0-30 hours between meals (she's very dramatic)
    hours_since_meal = torch.rand(n_observations, 1) * 30

    # The sacred relationship known to ancient cat scholars
    base_hunger = 20
    hunger_per_hour = 2.5

    hunger_levels = hunger_per_hour * hours_since_meal.squeeze() + base_hunger

    # Add feline chaos (cats are unpredictable creatures)
    chaos = torch.randn(n_observations) * chaos_level * hunger_levels.std()
    hunger_levels = hunger_levels + chaos

    # Even mystical cats have limits
    hunger_levels = torch.clamp(hunger_levels, 0, 100)

    return hours_since_meal, hunger_levels.unsqueeze(1)

def visualize_cat_wisdom(hours: torch.Tensor, hunger: torch.Tensor,
                        predictions: torch.Tensor = None):
    """Display the sacred patterns of Suki's appetite."""
    plt.figure(figsize=(12, 7))
    plt.scatter(hours.numpy(), hunger.numpy(), alpha=0.6, color='purple',
                label='Suki\'s Actual Hunger Levels')

    if predictions is not None:
        sorted_indices = torch.argsort(hours.squeeze())
        sorted_hours = hours[sorted_indices]
        sorted_predictions = predictions[sorted_indices]
        plt.plot(sorted_hours.numpy(), sorted_predictions.detach().numpy(),
                'gold', linewidth=3, label='Your Mystical Predictions')

    plt.axhline(y=FEEDING_THRESHOLD, color='red', linestyle='--', alpha=0.7,
                label='Sacred Feeding Threshold (Suki Appears!)')
    plt.xlabel('Hours Since Last Meal (feature)')
    plt.ylabel('Suki\'s Hunger Level (target)')
    plt.title('The Mysteries of Temple Cat Appetite')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.show()
```

### 4. Starter Code Template
- PyTorch class with themed name
- Clear TODO comments with hints
- Training function with gradient management
- Proper error handling and progress reporting
- Use industry standard function and variable names
    - Prefer `features` over `x` and `target` over `y`
    - AVOID mystical/temple-themed names

**Example Template:**

```python
# 💃 FIRST MOVEMENTS

class CatHungerPredictor(nn.Module):
    """A mystical artifact for understanding feline appetite patterns."""

    def __init__(self, input_features: int = 1):
        super(CatHungerPredictor, self).__init__()
        # TODO: Create the Linear layer
        # Hint: torch.nn.Linear transforms input energy into output wisdom
        # It needs input_features and output_features (how many predictions?)
        self.linear = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Channel your understanding through the mystical network."""
        # TODO: Pass the input through your Linear layer
        # Remember: even cats follow mathematical laws
        return None

def train(model: nn.Module, features: torch.Tensor, target: torch.Tensor, epochs: int = 4_000) -> list:
    """
    Train the cat hunger prediction model.

    Returns:
        List of loss values during training
    """
    # TODO: Choose your loss calculation method
    # Hint: Mean Squared Error is favored by the ancient masters
    criterion = None

    # TODO: Choose your parameter updating method
    # Hint: SGD (Stochastic Gradient Descent) is the traditional path
    optimizer = None

    losses = []

    gradient_is_good = False
    previous_loss = None
    for epoch in range(epochs):
        # TODO: CRITICAL - Clear the gradient spirits from previous cycle
        # Hint: The spirits accumulate if not banished properly
        # This is the most common mistake in PyTorch training!

        # TODO: Forward pass - get predictions
        predictions = None

        # TODO: Compute the loss
        loss = None

        # TODO: Backward pass - compute gradients
        # Hint: Loss knows how to compute its own gradients

        # TODO: Update parameters
        # Hint: The optimizer knows how to update using the gradients

        losses.append(loss.item())

        # Report progress to Master Pai-Torch
        if (epoch + 1) % int(epochs / 10) == 0:
            gradient_message = ""
            # Stable enough gradient
            if not gradient_is_good and previous_loss and 1 - (loss / previous_loss) <= 0.01:
                gradient_message = " 💫 The Gradient Spirits smile upon your progress!"
                gradient_is_good = True
            previous_loss = loss
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f} {gradient_message}')

    return losses
```

### 5. Success Criteria
- "⚡ THE TRIALS OF MASTERY" section
- Dynamic checkbox system with real-time evaluation
- Quantitative metrics with pass/fail indicators
- Comprehensive performance analysis
- Clear success and failure messaging

**Example Template:**

```python
# TRIALS OF MASTERY
print("⚡ TRIAL 1: BASIC MASTERY")

# Check your progress
final_loss = loss_history[-1] if loss_history else float('inf')
weight_accuracy = abs(learned_weight - 2.5) < 0.5
bias_accuracy = abs(learned_bias - 20) < 5

# Check if loss decreases consistently (last loss < first loss by significant margin)
loss_decreases = len(loss_history) > 100 and loss_history[-1] < loss_history[99] * 0.9

# Check if predictions form a clean line (R² > 0.8)
with torch.no_grad():
    predictions = model(hours_since_meal)
    y_mean = hunger_levels.mean()
    ss_tot = ((hunger_levels - y_mean) ** 2).sum()
    ss_res = ((hunger_levels - predictions) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot)
    clean_line = r_squared > 0.8

# Trial 1 checkboxes
loss_check = "✅" if loss_decreases else "❌"
weight_bias_check = "✅" if (weight_accuracy and bias_accuracy) else "❌"
line_check = "✅" if clean_line else "❌"

print(f"- {loss_check} Loss decreases consistently (no angry Gradient Spirits)")
print(f"- {weight_bias_check} Model weight approximately 2.5 (±0.5), bias around 20 (±5)")
print(f"- {line_check} Predictions form a clean line through the scattered data")

# Trial 2: Understanding Test
print("\n⚡ TRIAL 2: UNDERSTANDING TEST")

# Test prediction shapes
test_features = torch.tensor([[5.0], [10.0], [20.0]])
with torch.no_grad():
    test_predictions = model(test_features)

shapes_correct = test_predictions.shape == (3, 1)
weight_reasonable = 2.0 <= learned_weight <= 3.0
bias_reasonable = 15 <= learned_bias <= 25

# Test prediction reasonableness
test_pred_values = test_predictions.squeeze().tolist()
expected_approx = [2.5 * 5 + 20, 2.5 * 10 + 20, 2.5 * 20 + 20]  # [32.5, 45, 70]
predictions_reasonable = all(abs(pred - exp) <= 10 for pred, exp in zip(test_pred_values, expected_approx))

# Trial 2 checkboxes
shapes_check = "✅" if shapes_correct else "❌"
weight_param_check = "✅" if weight_reasonable else "❌"
bias_param_check = "✅" if bias_reasonable else "❌"
pred_check = "✅" if predictions_reasonable else "❌"

print(f"- {shapes_check} Tensor shapes align with the sacred geometry")
print(f"- {weight_param_check} Weight parameter reflects feline wisdom")
print(f"- {bias_param_check} Bias parameter captures base hunger levels")
print(f"- {pred_check} Predictions are reasonable for test inputs")

# Your Performance section
print(f"\n📊 Your Performance:")
print(f"- Weight accuracy: {learned_weight:.3f} {'(PASS)' if weight_accuracy else '(FAIL)'}")
print(f"- Bias accuracy: {learned_bias:.3f} {'(PASS)' if bias_accuracy else '(FAIL)'}")

# Overall success check
trial1_passed = loss_decreases and weight_accuracy and bias_accuracy and clean_line
trial2_passed = shapes_correct and weight_reasonable and bias_reasonable and predictions_reasonable

if trial1_passed and trial2_passed:
    print("\n🎉 Master Pai-Torch nods with approval - your understanding grows!")
    print("\n🏆 Congratulations! You have passed the basic trials of the Temple Sweeper!")
    print("🐱 Suki purrs approvingly - your neural network has learned her sacred patterns.")
else:
    print("\n🤔 The path to mastery requires more practice. Consider adjusting your training parameters.")
    print("💡 Hint: Try different learning rates, more epochs, or review your code for errors.")
```

### 6. Progressive Extensions (4 EXTENSIONS REQUIRED)
- "🌸 THE FOUR PATHS OF MASTERY: PROGRESSIVE EXTENSIONS"
- Simple exploratory bullet points with emojis for visual appeal
- Focus on encouraging experimentation rather than full implementations
- Each suggestion should build understanding through hands-on exploration

**Example Template:**

```markdown
## 🌸 THE FOUR PATHS OF MASTERY: PROGRESSIVE EXTENSIONS

*Master Pai-Torch gestures toward four different pathways leading deeper into the temple.*

"You have learned the fundamental way, grasshopper. But mastery comes through exploring the branching paths."

🔍 **Reduce the number of epochs.** How well does the model fit?
⚡ **Increase or decrease the learning rate in SGD (default is 0.001).** What happens to the loss? What if you adjust the number of epochs? Can you make it converge?
🎯 **Increase the chaos in Suki's data.** How chaotic can you make it and still get reasonable results?
🌟 **Increase or decrease the number of observations in Suki's data.** What's the minimum amount needed for learning? What happens if you increase it? Does it affect the required number of epochs or the learning rate?
```

### 7. Completion Ceremony
- "🏆 COMPLETION CEREMONY" section
- Summary of sacred knowledge acquired
- Final wisdom and encouragement
- Reference to next steps in their journey

**Example Template:**

```markdown
## 🏆 COMPLETION CEREMONY

*Master Pai-Torch rises and bows respectfully*

"Congratulations, young grasshopper. You have successfully completed your first kata in the Temple of Neural Networks. Through Suki's simple feeding patterns, you have learned the fundamental mysteries that underlie all neural arts:

**Sacred Knowledge Acquired:**
- **Tensor Mastery**: You can create and manipulate PyTorch tensors with confidence
- **Linear Wisdom**: You understand how neural networks transform input to output
- **Gradient Discipline**: You have mastered the sacred training loop and gradient management
- **Loss Understanding**: You can measure and minimize prediction errors

**Final Wisdom:**
Remember always: every complex neural network, no matter how sophisticated, is built upon the simple principles you practiced here. The gradient flows, the loss decreases, and wisdom emerges from the dance between prediction and reality.

🐱 *Suki purrs approvingly from her perch, as if to say: "You are ready for greater challenges, young neural warrior."*

🏮 **May your gradients flow smoothly and your losses converge swiftly!** 🏮"
```

## Technical Requirements
- All code must use PyTorch (even for simple problems)
- **All imports, global configurations, and constants must be consolidated in the first code cell**
  - Include all necessary imports (torch, torch.nn, torch.optim, matplotlib.pyplot, etc.)
  - Define any global constants or configuration variables
  - Include welcoming print statements with temple theming
  - This ensures clean notebook structure and easy dependency management
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
- Prefix the name with "kata_nn_" with the kata number. Future katas could use concepts from all previous katas and dans.
- Example: `dan_1/kata_01_temple_cat_feeding_predictor_unrevised.ipynb`

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
