# PyTorch Katas: Claude Code Operations Guide

## Project Purpose & Mission

This repository is a **PyTorch Code Kata** collection designed to build "muscle memory" and intuition for neural networks through deliberate practice. The core philosophy is **practice-focused learning** - learners practice PyTorch concepts they're studying elsewhere, reinforcing knowledge through hands-on coding exercises.

### Key Principles:
- **Practice, not teaching** - assumes learners have basic familiarity with concepts
- **PyTorch for everything** - even simple problems use neural networks to build PyTorch fluency
- **Progressive skill building** - Dan levels (1-5) increase in complexity
- **Engaging narrative** - temple setting with memorable characters makes learning enjoyable

## Repository Structure

```
pytorch-katas/
├── dan_1/          # Temple Sweeper: Basic tensors, linear models, training loops
├── dan_2/          # Temple Guardian: Regularization, validation, optimization
├── dan_3/          # Weapon Master: CNNs, RNNs, attention mechanisms
├── dan_4/          # Combat Innovator: Custom losses, multi-task learning
├── dan_5/          # Mystic Arts Master: GANs, diffusion models, meta-learning
├── CLAUDE.md       # This operational guide
├── README.md       # Public-facing project description
└── .claude/
    └── commands/
        └── create-kata.md  # Detailed kata creation instructions
```

### File Naming Convention:
- **Format**: `kata_XX_descriptive_name_unrevised.ipynb`
- **Example**: `kata_01_temple_cat_feeding_predictor_unrevised.ipynb`
- **Suffix**: All new katas get `_unrevised` until human review

## Temple Characters & Usage Guidelines

### Character Personalities (Brief Reference):
- **Master Pai-Torch**: Cryptic koans about gradients, appears when learners are stuck
- **Master Ao-Tougrad**: Mysterious backpropagation keeper, leaves helpful hints
- **Cook Oh-Pai-Timizer**: Relates cooking to optimization, practical hands-on approach
- **He-Ao-World**: Apologetic janitor, introduces real-world data problems through "accidents"
- **Suki**: Sacred temple cat, behaviors serve as training data

### Character Usage Rules:
- **Gender-neutral language** - never use "he/she" except for Suki (female cat)
- **Address learner as "you" or "Grasshopper"**
- **Authentic personalities** - each character should feel consistent with their established persona
- **Natural integration** - problems should emerge organically from character expertise

### He-Ao-World: The Hidden Master (Secret Lore)
**Background**: He-Ao follows the "hidden master" trope. Years of observing temple masters while cleaning gave He-Ao deep wisdom about neural networks. He-Ao learned through practical application - bar fights, bandit encounters, real-world scenarios. The masters are unaware of He-Ao's true skills.

**Narrative Purpose**: Introduces real-world data problems (class imbalance, missing data, noise) through well-intentioned "accidents" that create messy, realistic datasets. Always apologetic but timing is suspiciously convenient.

## Coding Standards & Practices

### PyTorch Conventions:
- **Use proper imports**: `import torch`, `import torch.nn as nn`, `import torch.optim as optim`
- **Reproducibility**: Always include `torch.manual_seed(42)` for consistent results
- **Standard variable names**: Prefer `features` over `x`, `target` over `y`
- **Industry patterns**: Follow PyTorch best practices for model classes and training loops

### Naming Convention Guidelines:

**CRITICAL DISTINCTION**: Data generation can be thematic, PyTorch templates must use industry standards.

#### ✅ Data Generation Functions - THEMATIC NAMES ALLOWED:
- **Function names**: `generate_cat_feeding_data()`, `create_temple_scrolls()`, `simulate_candle_burning()`
- **Parameters**: `chaos_level`, `sacred_seed`, `temple_cats`, `scroll_authenticity`
- **Return variables**: `hours_since_meal`, `hunger_levels`, `scroll_features`, `authenticity_labels`
- **Visualization**: `visualize_cat_wisdom()`, `plot_temple_mysteries()`

#### ✅ PyTorch Model Templates - INDUSTRY STANDARD NAMES REQUIRED:
- **Function parameters**: `features` (not `X`), `target` (not `y`)
- **Model methods**: `forward()` (not `divine_hunger()` or similar)
- **Training variables**: `predictions`, `loss`, `optimizer`, `criterion`
- **Model attributes**: `self.linear`, `self.conv1`, `self.dropout` (not `self.linear_wisdom`)

#### ❌ AVOID in PyTorch Templates:
- Mystical parameter names: `divine_hunger()`, `sacred_features`, `mystical_predictions`
- Temple-themed variables: `gradient_spirits`, `loss_wisdom`, `tensor_magic`
- Non-standard abbreviations: `X`, `y`, `pred`, `opt`

**Example of Correct Usage:**
```python
# Data generation - thematic names OK
hours_since_meal, hunger_levels = generate_cat_feeding_data(chaos_level=0.1)

# PyTorch model - industry standard names required
def train(model, features, target, epochs=1000):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        predictions = model(features)  # Not: mystical_output = model.divine_hunger(features)
        loss = criterion(predictions, target)
```

### Code Quality Requirements:
- **Complete working examples** - all code should run without errors
- **Clear documentation** - docstrings for all functions and classes
- **Proper error handling** - graceful handling of edge cases
- **Visualization included** - matplotlib plots for data understanding
- **No external dependencies** - generate synthetic data only

### Anti-Patterns to Avoid:
- **Mystical variable names** - avoid temple-themed names in actual code
- **Incomplete TODOs** - provide clear hints and guidance
- **Missing gradient management** - always include `optimizer.zero_grad()`
- **Inconsistent difficulty** - ensure progressive learning within dan levels

## Dan Level Specifications

| Dan | Title | Focus Areas | Typical Concepts |
|-----|-------|-------------|------------------|
| 1 | Temple Sweeper | Foundation | Tensors, linear layers, basic training, gradient descent |
| 2 | Temple Guardian | Robustness | Regularization, validation, multiple layers, optimizers |
| 3 | Weapon Master | Specialization | CNNs, RNNs, attention, transfer learning |
| 4 | Combat Innovator | Innovation | Custom losses, multi-task, adversarial training |
| 5 | Mystic Arts Master | Generation | GANs, diffusion, graph networks, meta-learning |

## Tools & Workflows

### Primary Tools:
- **Jupyter Notebooks** - all katas delivered as `.ipynb` files
- **Google Colab** - include badge for easy cloud access
- **PyTorch** - core framework for all exercises
- **Matplotlib** - visualization and plotting
- **NumPy** - supporting numerical operations

### Creation Workflow:
1. **Check existing katas** - read README.md to avoid duplication
2. **Use create-kata command** - follow structured template
3. **Add to README.md** - update kata index with new entry
4. **Test thoroughly** - ensure all code runs correctly
5. **Mark as unrevised** - human review required before finalization

## Common Patterns & Solutions

### Synthetic Data Generation:
- **Themed functions** - match temple setting (e.g., `generate_cat_feeding_data`)
- **Controllable difficulty** - parameters for chaos/noise levels
- **Visualization included** - always provide plotting functions
- **Real-world flavoring** - He-Ao-World introduces realistic data problems

### Training Loop Template:
```python
def train(model, features, target, epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()        # Critical: clear gradients
        predictions = model(features)
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.4f}')
    
    return model
```

### Success Criteria Pattern:
- **Quantitative metrics** - specific loss thresholds, accuracy targets
- **Qualitative checks** - convergence behavior, visualization quality
- **Parameter validation** - ensure learned weights make sense
- **Unit tests** - `test_your_wisdom()` function for validation

## Lessons Learned & Common Corrections

### Frequent Issues:
- **Missing `optimizer.zero_grad()`** - leads to gradient accumulation
- **Incorrect tensor shapes** - ensure proper dimensionality
- **Hardcoded values** - use configurable parameters
- **Poor visualization** - include informative plots and legends

### Best Practices Discovered:
- **Progressive difficulty** - each extension should build naturally
- **Character consistency** - maintain authentic personalities
- **Clear learning objectives** - specific, measurable goals
- **Debugging sections** - include intentional mistakes for learning

## User Preferences & Patterns

### Preferred Learning Style:
- **Learning by doing** - hands-on coding over theoretical explanations
- **Engaging narratives** - story-driven problems maintain interest
- **Progressive challenge** - gradual difficulty increase within dan levels
- **Practical relevance** - real-world data problems through He-Ao-World

### Common Requests:
- **More complex scenarios** - users enjoy challenging extensions
- **Debugging practice** - intentional mistakes for skill building
- **Character interactions** - multiple characters in single kata
- **Visualization quality** - clear, informative plots

## Integration Notes

### README.md Updates:
- **Always update** - add new katas to the index
- **Consistent formatting** - follow existing style patterns
- **Clear descriptions** - brief but informative kata summaries
- **Proper links** - ensure file paths are correct

### Quality Assurance:
- **Code testing** - all examples must run without errors
- **Character authenticity** - maintain established personalities
- **Difficulty progression** - ensure appropriate challenge level
- **Learning objectives** - clear, achievable goals
