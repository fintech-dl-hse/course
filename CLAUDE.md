# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Deep Learning course repository for the Fintech faculty at HSE University. The repository contains lecture materials, seminar materials, homework assignments, and educational videos/animations. The course covers topics including neural network fundamentals, computer vision, generative models, NLP, transformers, and large language models.

## Repository Structure

- **`lectures/`** - Lecture materials (currently minimal, content added throughout the course)
- **`seminars/`** - Seminar materials (currently minimal, content added throughout the course)
- **`videos/`** - Video files and animations for educational purposes, including Manim-generated content
- **`.public/`** - Static files deployed to GitHub Pages (homework links, etc.)
- **`.manim/`** - Manim cache directory
- **`custom_config.yml`** - Manim configuration file

## Key Technologies

## Development Commands

### Jupyter Notebooks

#### Jupyter Path

**CRITICAL**: Always use the correct Jupyter path for this environment:

```bash
~/miniconda3/envs/audio/bin/jupyter
```

#### Creating Jupyter Notebooks Programmatically

When creating or modifying Jupyter notebooks via Python scripts, **ALWAYS** ensure proper cell formatting:

**CORRECT ✅ - Each line is a separate array element:**

```python
def create_code_cell(code: str) -> dict:
    """Create a properly formatted code cell."""
    lines = code.strip().split('\n')
    source = [line + '\n' for line in lines[:-1]]
    if lines:
        source.append(lines[-1])  # Last line without \n

    return {
        "cell_type": "code",
        "execution_count": None,
        "source": source,  # Array of strings!
        "outputs": [],
        "metadata": {}
    }
```

**INCORRECT ❌ - Code as single string:**

```python
# DON'T DO THIS!
cell = {
    "cell_type": "code",
    "source": "import torch\nimport numpy as np\n..."  # Single string!
}
```

**Why this matters:**
- Jupyter requires each line as a separate array element
- Single-string source won't display properly in Jupyter
- Code will appear as one unreadable line
- Copy-paste from created notebooks will fail

**Example of properly formatted cell source:**

```json
{
  "cell_type": "code",
  "source": [
    "import torch\n",
    "import numpy as np\n",
    "\n",
    "def train(model):\n",
    "    # Training code\n",
    "    pass"
  ]
}
```

#### Jupyter Commands

```bash
# Convert to Python script
~/miniconda3/envs/audio/bin/jupyter nbconvert --to script <filename>.ipynb

# Clear outputs
~/miniconda3/envs/audio/bin/jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace <filename>.ipynb
```

## Deployment

The repository uses GitHub Actions to deploy static content to GitHub Pages:
- Workflow: `.github/workflows/deploy-pages.yml`
- Triggered on push to `main` branch or manual dispatch
- Deploys contents of `.public/` directory
- Remote: `git@github.com:fintech-dl-hse/course.git`

## Course Structure Notes

- The course runs across multiple years (branches: 2022, 2023, 2024, 2025)
- Main branch contains the core course structure
- Content is added incrementally throughout the course
- Homework assignments are distributed and graded via GitHub Classroom
- Students submit homework through automated grading pipelines
- Late submissions incur a 10% penalty per day (max 30%)

## Important Patterns

- This repository is primarily content-focused (educational materials)
- Most code is in the form of Manim animations for visualization
- The repository structure is deliberately minimal to be populated during the course
- Static HTML files in `.public/` serve as redirects to external resources (e.g., Yandex Cloud Functions)

## Seminar Materials Structure

### Naming Conventions

- Active seminars: `01_seminar_<topic>.ipynb`, `02_seminar_<topic>.ipynb`, etc.
- Helper scripts: `merge_notebooks.py`, `rebuild_notebook.py`, `test_notebook.py`

### Seminar Notebook Pattern

### Pedagogical Approach

**"Outside-In" principle:**
1. First, teach students to **USE** the tools (PyTorch, libraries)
2. Then, explain **HOW THEY WORK** internally (algorithms, math)
3. Finally, **PRACTICE** with hands-on exercises

This approach helps students:
- Build confidence with working code first
- Understand "why" after seeing "what"
- Solidify learning through practice

### Code Quality Standards for Seminars

**Required:**
- Type hints on all function signatures
- Docstrings with Args and Returns
- Inline comments explaining non-obvious logic
- Modern Python style (f-strings, type annotations)

**Example:**

```python
def train(model: nn.Module,
          learning_rate: float = 0.1,
          num_steps: int = 300) -> None:
    """
    Обучает модель используя ручной SGD.

    Args:
        model: Модель для обучения
        learning_rate: Скорость обучения
        num_steps: Количество шагов оптимизации
    """
    # Implementation...
```

## Common Pitfalls to Avoid

### Jupyter Notebooks

❌ **DON'T:**
- Create cell source as a single string
- Forget to clear outputs before committing
- Use wrong Jupyter path (must use `~/miniconda3/envs/audio/bin/jupyter`)
- Edit notebook JSON manually
- Commit notebooks with large outputs (>1 MB)

✅ **DO:**
- Use proper cell formatting (array of strings with `\n`)
- Always clear outputs before committing
- Test notebooks with "Restart & Run All"
- Use helper scripts for notebook manipulation
- Keep notebooks under 100 KB (without outputs)

### Code Style

❌ **DON'T:**
- Skip type hints on educational code
- Use unclear variable names
- Omit docstrings on key functions
- Write overly complex code for teaching

✅ **DO:**
- Add type hints everywhere
- Use clear, descriptive names
- Write comprehensive docstrings
- Prioritize readability over cleverness
- Add comments explaining "why", not just "what"

### Repository Organization

❌ **DON'T:**
- Leave work-in-progress files uncommitted for long periods
- Create notebooks without updating README
- Delete old materials without archiving (prefix with `old_`)
- Mix different course years in main branch

✅ **DO:**
- Archive old materials with `old_` prefix
- Update README.md when adding new seminars
- Keep different years in separate branches
- Maintain clean main branch structure

## Seminars Overview

### Core themes we cover (use as anchors)
- In each seminar, avoid topics that students have not covered yet; use the plan implied by the list below. For example, do not use CNNs in seminars until the CV block has been covered.
- Training basics: backprop, losses, activations, weight init.
- Optimization + regularization: SGD/Adam/Muon, LR schedules, weight decay, dropout.
- CV: CNNs, vision tasks; later ViT + self-supervised/contrastive.
- Generative: AR, GAN, VAE, diffusion.
- NLP: word2vec, tokenization (BPE/WordPiece/SentencePiece), RNN/attention/transformers, pretrained LMs.
- LLM engineering: scaling laws, in-context learning, test-time scaling, PEFT.
- Agents & systems: function calling, agentic patterns, observability, MCP, RAG.
- Multimodal models.

### Seminar crafting style (lightweight)
- One concept per cell/section; keep cells short and runnable top-to-bottom.
- Prefer clarity over cleverness: explicit names, small helper functions, minimal magic.
- Add type hints + brief docstrings for reusable helpers.
- Jupyter math: use `$...$` for inline math (not `\(...\)`), and `$$...$$` for display math on separate lines (not `\[...\]`).
- Questions in seminars: one question per separate cell; format as a level-4 heading with ❓ and bold **Вопрос**, e.g. `#### ❓ **Вопрос**: зачем делать Softmax? Ведь на выходе модельки мы уже получаем логиты?`
- Answers on questions: all answers must be hidden by spoilers. Example:
```md
<details>

<summary><strong>Ответ</strong></summary>

Текст ответа</br>
Следующая строчка ответа</br>

</details>
```
- All questions in section `Блиц` must have answers hidden by spoilers. Add them if there is no answer yet.
- Make experiments reproducible: fixed seeds, deterministic settings (when feasible), explicit dependencies.
- Before committing notebooks: clear outputs; avoid huge artifacts in repo.
- Do not run `./seminars/rm_widgets.sh` manually: it is executed automatically.
- No trailing spaces.
- First Cell must be Heading 1 title of seminar (# MLP)
