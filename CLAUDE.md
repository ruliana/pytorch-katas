# PyTorch Katas

This is a Github repository for PyTorch Code Katas.

The goal is to provide exercises for the user to build "muscle memory" and intuition about Neural Networks and PyTorch.

Below are instructions about how to create new katas and the setting to make then engaging.

## The setting

The setting in the @README.md pictures a journey from a disciple to master. Below is the hidden story of the characters and instructions on how to use them.

### Guidelines

- The characters shouldn't imply they are of any gender (let that for the reader's imagination), except for Suki (a female cat). The masters, the cook, and the janitor are always referred by their names, never with `he` or `she`. The grasshopper is always addressed as `you`.

### He-Ao-World

He-Ao follows the trope of the hidden master. From the years in the temple, He-Ao learned by observing the masters and applying that in real life, in bar fights, fighting against bandits and so on. The masters are unaware of how skillful is He-Ao and the old janitor sympathizes and want to help the humble grasshopper to achieve true greatness.

He-Ao-World should be used to introduce problems in data that we are likely to find in real world: class imbalance, missing data, noise, and so on.

## Kata Structure Requirements

Every kata must include these components:

### 1. Header Section
- Title: Descriptive name
- Dan Level: 1-5 (difficulty progression)
- Concepts: List of 3-4 PyTorch concepts being practiced
- Time Estimate: Realistic completion time

Example:

```
🏮 The Ancient Scroll Unfurls 🏮

THE TRIAL OF THE TEMPLE CAT'S APPETITE
Dan Level: 1 (Temple Sweeper) | Time: 45 minutes | Sacred Arts: Tensor Flows, Linear Wisdom, Training Rituals
```

### 2. Problem Description
- Clear, engaging problem statement
- Specific learning objectives

Example:

```
📜 THE MASTER'S CHALLENGE

Young Grasshopper, your first trial has arrived with the morning mist.

Suki, our sacred temple cat, has blessed us with her presence for many moons.
Yet her feeding patterns remain a mystery that has puzzled temple novices. She
arrives at her food bowl seemingly at random, leaving offerings to grow stale
and disrupting the temple's harmony.

*CRASH!*

"Oh! So sorry!" calls He-Ao-World from across the courtyard, where he's just knocked
over a small pile of feeding time scrolls. "These old hands! I was just trying
to organize the records and... well, now some are a bit mixed up. Hope that's
not important for your training!"

"The cat," whispers Master Pai-Torch from behind his meditation cushion,
seemingly unbothered by the commotion, "follows patterns invisible to the
untrained eye. Hours since her last meal flow like water down the mountain -
predictable to those who understand the current."

Your sacred duty: Create a linear model that can predict when Suki will grace us with
her presence at the food bowl.

🎯 THE SACRED OBJECTIVES

- [ ] Master the creation of data tensors
- [ ] Forge your first Linear Wisdom artifact using torch.nn.Linear
- [ ] Perform the Training Ritual: forward pass → loss computation → backpropagation → parameter update
- [ ] Observe the mystical loss decreasing over time (if it increases, you have angered the Gradient Spirits)
```

### 3. Synthetic Dataset Generator
- Python function that creates the dataset
- Controllable parameters for difficulty scaling
- Include data visualization code
- No external data dependencies

Example:

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
- Minimal scaffolding with clear TODOs
- Import statements and basic structure
- Helper functions if needed
- Clear comments indicating what to implement

Example:

```python
# 💃 FIRST MOVEMENTS

class CatHungerPredictor(nn.Module):
    """A mystical artifact for understanding feline appetite patterns."""

    def __init__(self, input_features: int = 1):
        super(CatHungerPredictor, self).__init__()
        # TODO: Create the Linear Wisdom layer
        # Hint: torch.nn.Linear transforms input energy into output wisdom
        self.linear_wisdom = None

    def divine_hunger(self, hours_fasting: torch.Tensor) -> torch.Tensor:
        """Channel your understanding through the mystical network."""
        # TODO: Pass the input through your Linear Wisdom
        # Remember: even cats follow mathematical laws
        return None

def train_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
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
- Quantitative metrics (accuracy thresholds, loss targets)
- Qualitative checks (convergence behavior, visualization requirements)
- Unit tests for key functions

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
    test_hours = torch.tensor([[5.0], [10.0], [20.0]])
    predictions = model(test_hours)
    assert predictions.shape == (3, 1), "The shapes must align!"

    # Parameters should reflect the true cat nature
    weight = model.linear.weight.item()
    bias = model.linear.bias.item()
    assert 2.0 <= weight <= 3.0, f"Weight {weight:.2f} seems off - cats are more predictable!"
    assert 15 <= bias <= 25, f"Bias {bias:.2f} - even well-fed cats have base hunger!"

    print("🎉 Master Pai-Torch nods with approval - your understanding grows!")
```

### 6. Progressive Extensions
- 3-4 increasingly difficult variants (use the characters to introduce them)
- Each extension teaches additional concepts
- Clear progression from basic to advanced

Example:

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

def normalize_feeding_data(hours: torch.Tensor, hunger: torch.Tensor):
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

def patient_training_ritual(model, X, y, epochs=2000, patience=100):
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

def analyze_feeding_threshold(model, X, y, threshold_candidates=[60, 65, 70, 75, 80]):
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

def visualize_decision_boundary(model, X, y, best_threshold):
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
- Common mistake intentionally introduced
- Hints for identification and fixing
- Teaches debugging skills and common pitfalls

Example:

```python
🔥 CORRECTING YOUR FORM: A STANCE IMBALANCE

Master Pai-Torch observes your training ritual with a careful eye. "Your eager mind races ahead of your disciplined form, grasshopper. See how your gradient flow stance wavers?"

A previous disciple left this flawed training ritual. Your form has become unsteady - can you restore proper technique?

```python
def unsteady_training_ritual(model, X, y, epochs=1000):
    """This training stance has lost its balance - your form needs correction! 🥋"""
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
            # Forward pass
        predictions = model(X)
            loss = criterion(predictions, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                    print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

            return model
```

## Organization

- There are 5 folders: `dan_1` to `dan_5`. Katas should be placed in the correct dan folder.
- Each kata should be a Jupyter Notebook (`.pynb`) with the text correctly interleaved with Python code to tell the story and present the problem.
- Each Jupyter Notebook should have a badge to open it in Google Colab
