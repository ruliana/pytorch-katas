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

### 2. Problem Description - DYNAMIC STORY GENERATION
- Start with "## 📜 THE CHALLENGE"
- **USE ORACLE TABLES** to generate unique story combinations (see Story Generation System below)
- Clear learning objectives with checkboxes using "🎯 THE SACRED OBJECTIVES"

## 🎲 STORY GENERATION SYSTEM (MANDATORY)

**STEP 0: Choose the Neural Network and PyTorch techniques**

Decide which technique will be the theme of the Kata. Carefully consider the Dan level and previous Katas.

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

1. **The Hidden Technique** - The Grasshopper discovers a secret technique through observation or accident
2. **The Shortcut's Punishment** - The Grasshopper attempts shortcuts in training and faces severe consequences/harder training
3. **Training Hidden as Mundane Tasks** - The Grasshopper is tasked with mundane chores that secretly teach advanced techniques
4. **The Forbidden Knowledge** - Dangerous technique must be mastered safely under strict supervision
5. **The Rival's Challenge** - A rival appears to challenge the Grasshopper's progress, forcing them to prove their mastery
6. **The Breaking Point** - The Grasshopper must overcome physical/mental limits through extreme conditioning
7. **The Master's Test** - The Grasshopper faces a seemingly impossible task that reveals their true understanding
8. **The Blindfolded Training** - Training while deprived of sight to develop other senses and intuition
9. **The Ancient Manual** - A cryptic training scroll must be deciphered and its techniques mastered
10. **The Final Confrontation** - The ultimate test where the Grasshopper must face their greatest challenge using all learned skills

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

**Example Template:**

```python
# 🐱 THE SACRED DATA GENERATION SCROLL

def generate_cat_feeding_data(n_observations: int = 100, chaos_level: float = 0.1,
                             sacred_seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
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
    torch.manual_seed(seed)

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

    plt.axhline(y=70, color='red', linestyle='--', alpha=0.7,
                label='Sacred Feeding Threshold (Suki Appears!)')
    plt.xlabel('Hours Since Last Meal')
    plt.ylabel('Suki\'s Hunger Level')
    plt.title('🐱 The Mysteries of Temple Cat Appetite 🐱')
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
        self.linear = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Channel your understanding through the mystical network."""
        # TODO: Pass the input through your Linear layer
        # Remember: even cats follow mathematical laws
        return None

def train(model: nn.Module, features: torch.Tensor, target: torch.Tensor,
               epochs: int = 1000, learning_rate: float = 0.01) -> list:
    """
    Train the cat hunger prediction model.

    Returns:
        List of loss values during training
    """
    # TODO: Choose your loss calculation method
    # Hint: Mean Squared Error is favored by the ancient masters
    criterion = None

    # TODO: Choose your parameter updating method
    # Hint: SGD is the traditional path, simple and effective
    optimizer = None

    losses = []

    for epoch in range(epochs):
        # TODO: CRITICAL - Clear the gradient spirits from previous cycle
        # Hint: The spirits accumulate if not banished properly

        # TODO: Forward pass - get predictions
        predictions = None

        # TODO: Compute the loss
        loss = None

        # TODO: Backward pass - compute gradients

        # TODO: Update parameters

        losses.append(loss.item())

        # Report progress to Master Pai-Torch
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            if loss.item() < 10:
                print("💫 The Gradient Spirits smile upon your progress!")

    return losses
```

### 5. Success Criteria
- "⚡ THE TRIALS OF MASTERY" section
- Quantitative metrics (loss targets, accuracy thresholds)
- Unit test function called "test_your_wisdom"
- Parameter validation matching expected values
- Convergence behavior requirements

**Example Template:**

```python
# ⚡ THE TRIALS OF MASTERY

## Trial 1: Basic Mastery
# - [ ] Loss decreases consistently (no angry Gradient Spirits)
# - [ ] Final loss below 50 (Suki approves of your predictions)
# - [ ] Model weight approximately 2.5 (±0.5), bias around 20 (±5)
# - [ ] Predictions form a clean line through the scattered data

## Trial 2: Understanding Test
def test_your_wisdom(model):
    """Master PyTorch's evaluation of your understanding."""
    # Your model should produce correct shapes
    test_features = torch.tensor([[5.0], [10.0], [20.0]])
    predictions = model(test_features)
    assert predictions.shape == (3, 1), "The shapes must align!"

    # Parameters should reflect the true cat nature
    weight = model.linear.weight.item()
    bias = model.linear.bias.item()
    assert 2.0 <= weight <= 3.0, f"Weight {weight:.2f} seems off - cats are more predictable!"
    assert 15 <= bias <= 25, f"Bias {bias:.2f} - even well-fed cats have base hunger!"

    print("🎉 Master Pai-Torch nods with approval - your understanding grows!")
```

### 6. Progressive Extensions (4 EXTENSIONS REQUIRED)
- "🌸 THE FOUR PATHS OF MASTERY: PROGRESSIVE EXTENSIONS"
- Each extension introduces a character with new challenges
- Progressive difficulty increase (+15%, +25%, +35%, +45%)
- New concepts for each extension
- Clear success criteria for each

**Example Template:**

```python
# 🌸 THE FOUR PATHS OF MASTERY: PROGRESSIVE EXTENSIONS

## Extension 1: Cook Oh-Pai-Timizer's Portion Control
# "A good cook knows that batch size affects the final dish!"

# *Cook Oh-Pai-Timizer bustles over, wooden spoon in hand*
#
# "Ah, grasshopper! I see you've mastered feeding one cat at a time. But what happens
# when you need to predict hunger for multiple cats simultaneously? In my kitchen,
# efficiency comes from preparing multiple servings at once!"

# NEW CONCEPTS: Batch processing, tensor shapes, vectorized operations
# DIFFICULTY: +15% (still Dan 1, but with batches)

def generate_multi_cat_data(n_cats: int = 5, observations_per_cat: int = 50):
    """
    Generate feeding data for multiple temple cats at once.
    
    Returns:
        Tuple of (batch_hours, batch_hunger_levels)
        Shape: (n_cats * observations_per_cat, 1) for both tensors
    """
    # TODO: Create batched data that your model can process all at once
    # Hint: Your existing model should work without changes!
    pass

# TRIAL: Feed batched data to your existing model
# SUCCESS: Model processes multiple cats simultaneously, same accuracy

## Extension 2: He-Ao-World's Measurement Mix-up
# "These old eyes sometimes read the measuring scrolls incorrectly..."

# *He-Ao-World shuffles over, looking apologetic*
#
# "Oh dear! I was recording Suki's feeding times and... well, I might have mixed up
# some of the measurements. Some are in minutes instead of hours, and others might
# be twice what they should be. The data looks a bit... chaotic now."

# NEW CONCEPTS: Data normalization, feature scaling, handling inconsistent units
# DIFFICULTY: +25% (still Dan 1, but messier data)

def normalize_feeding_data(hours_since_meal: torch.Tensor, hunger_levels: torch.Tensor):
    """
    Clean and normalize the feeding data to handle measurement inconsistencies.
    
    Returns:
        Tuple of (normalized_hours, normalized_hunger)
    """
    # TODO: Implement data normalization
    # Hint: (data - mean) / std is a common normalization approach
    # Remember: Store the normalization parameters for later use!
    pass

# TRIAL: Train your model on normalized data
# SUCCESS: Model converges faster and more reliably

## Extension 3: Master Pai-Torch's Patience Teaching
# "The eager student trains too quickly and learns too little."

# *Master Pai-Torch sits in contemplative silence*
#
# "Young grasshopper, I observe your training ritual rushes like a mountain stream.
# But wisdom comes to those who vary their pace. Sometimes we must step boldly,
# sometimes cautiously, sometimes we must rest entirely."

# NEW CONCEPTS: Learning rate scheduling, early stopping, training patience
# DIFFICULTY: +35% (still Dan 1, but smarter training)

def patient_training_ritual(model, features, target, epochs=2000, patience=100):
    """
    Train with patience and adaptive learning rate.
    
    Args:
        patience: Stop training if loss doesn't improve for this many epochs
    
    Returns:
        Tuple of (trained_model, loss_history, stopped_early)
    """
    # TODO: Implement patient training with learning rate decay
    # Hint: Start with lr=0.1, reduce by half every 500 epochs
    # Hint: Keep track of best loss and stop if no improvement
    pass

# TRIAL: Compare patient training vs. rushed training
# SUCCESS: Patient training achieves better final loss with fewer wasted epochs

## Extension 4: Suki's Feeding Threshold Mystery
# "Understanding when the cat appears is as important as predicting hunger."

# *Suki sits majestically, then meows once*
#
# *Master Pai-Torch translates: "The sacred cat says your linear wisdom is sound, but
# the true test is knowing when hunger becomes action. At what point does prediction
# become decision?"*

# NEW CONCEPTS: Threshold analysis, decision boundaries, model interpretation
# DIFFICULTY: +45% (still Dan 1, but thinking beyond prediction)

def analyze_feeding_threshold(model, features, target, threshold_candidates=[60, 65, 70, 75, 80]):
    """
    Analyze how well your model predicts when Suki will actually appear.
    
    Returns:
        Dictionary of {threshold: accuracy_score}
    """
    # TODO: For each threshold, calculate:
    # - How often model predicts "Suki will appear" (prediction > threshold)
    # - How often this prediction is correct
    # - Find the threshold that maximizes accuracy
    pass

def visualize_decision_boundary(model, features, target, best_threshold):
    """
    Show where your model draws the line between "hungry" and "will appear"
    """
    # TODO: Create a visualization showing:
    # - Original data points
    # - Model predictions
    # - Decision threshold line
    # - True/false positive regions
    pass

# TRIAL: Find the optimal threshold for predicting Suki's appearance
# SUCCESS: Achieve >80% accuracy in predicting when Suki will show up
# MASTERY: Understand that good predictions don't always mean good decisions
```

### 7. Debugging Challenge
- "🔥 CORRECTING YOUR FORM: A STANCE IMBALANCE" section
- Intentionally flawed code with common mistakes
- Master Pai-Torch providing guidance
- Clear hints for identification and fixing
- Focused on gradient/training issues

**Example Template:**

```python
# 🔥 CORRECTING YOUR FORM: A STANCE IMBALANCE

# Master Pai-Torch observes your training ritual with a careful eye. "Your eager mind races ahead of your disciplined form, grasshopper. See how your gradient flow stance wavers?"

# A previous disciple left this flawed training ritual. Your form has become unsteady - can you restore proper technique?

def unsteady_training(model, features, target, epochs=1000):
    """This training stance has lost its balance - your form needs correction! 🥋"""
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        # Forward pass
        predictions = model(features)
        loss = criterion(predictions, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

    return model

# DEBUGGING CHALLENGE: Can you spot the critical error in this training ritual?
# HINT: The Gradient Spirits are not being properly dismissed between cycles
# MASTER'S WISDOM: "The undisciplined mind accumulates old thoughts, just as the untrained gradient accumulates old directions."
```

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
