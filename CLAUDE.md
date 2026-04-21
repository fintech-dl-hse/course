# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Deep Learning course repository for the Fintech faculty at HSE University. The repository contains lecture materials, seminar materials, homework assignments, and educational videos/animations. The course covers topics including neural network fundamentals, computer vision, generative models, NLP, transformers, and large language models.

## Repository Structure

- **`lectures/`** - Lecture materials (currently minimal, content added throughout the course)
- **`seminars/`** - Seminar materials (currently minimal, content added throughout the course)
- **`.public/`** - Static files deployed to GitHub Pages (homework links, etc.)

## Development Commands

### Jupyter Notebooks

#### Jupyter Path

**CRITICAL**: Always use the correct Jupyter path for this environment:

```bash
~/miniconda3/envs/audio/bin/jupyter
```

## Deployment

The repository uses GitHub Actions to deploy static content to GitHub Pages:
- Workflow: `.github/workflows/deploy-pages.yml`
- Triggered on push to `main` branch or manual dispatch
- Deploys contents of `.public/` directory
- Remote: `git@github.com:fintech-dl-hse/course.git`

## Important Patterns

- This repository is primarily content-focused (educational materials)
- Most code is in the form of Manim animations for visualization
- The repository structure is deliberately minimal to be populated during the course
- Static HTML files in `.public/` serve as redirects to external resources (e.g., Yandex Cloud Functions)

## Seminar Materials Structure

### Pedagogical Approach

**"Outside-In" principle:**
1. First, teach students to **USE** the tools (PyTorch, libraries)
2. Then, explain **HOW THEY WORK** internally (algorithms, math)
3. Finally, **PRACTICE** with hands-on exercises

This approach helps students:
- Build confidence with working code first
- Understand "why" after seeing "what"
- Solidify learning through practice

---