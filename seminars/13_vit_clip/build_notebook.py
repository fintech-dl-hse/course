"""Generator for Seminar 13 notebook: Vision Transformers, Self-supervised, CLIP.

Run:
    ~/miniconda3/envs/audio/bin/python seminars/13_vit_clip/build_notebook.py

Produces 13_seminar_vit_clip.ipynb with properly formatted cells (each line is a
separate array element, as required by CLAUDE.md).

Note: the loss-scaling part of CLIP (logit_scale / temperature) is intentionally
left as an open question — students implement it themselves in the `clip` homework.
"""
import json
import os

CELLS = []

# Относительный путь до картинок-схем (рендерятся в локальном Jupyter и на GitHub).
IMG_BASE = "static"


def md(text: str) -> None:
    """Append a markdown cell from a raw string."""
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": _src(text)})


def code(text: str) -> None:
    """Append a code cell from a raw string."""
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(text),
    })


def _src(text: str):
    """Split text into a Jupyter source array (each line keeps its \\n)."""
    text = text.strip("\n")
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


# ---------------------------------------------------------------------------
# Cell 0: Colab badge + title + plan
# ---------------------------------------------------------------------------
md(
    '<a target="_blank" href="https://colab.research.google.com/github/fintech-dl-hse/course/blob/main/seminars/13_vit_clip/13_seminar_vit_clip.ipynb">\n'
    '  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>\n'
    "</a>"
)

md("# Vision Transformers. Self-supervised & Contrastive Learning")

md(
    """
*Это последний содержательный семинар курса. Хорошая новость: почти всё новое сегодня — это **перенос уже знакомых идей из NLP в компьютерное зрение**.*
"""
)

md("---")

md(
    """
## План семинара

Мы уже знаем трансформеры, self-attention, CLS-токен (из `BERT`), позиционные эмбэддинги и идею self-supervised предобучения в тексте (MLM). Сегодня применим всё это к картинкам.

1. **Recap** — мост из NLP-блока: что именно мы переносим в зрение.
2. **ViT (Vision Transformer)** — как скормить картинку трансформеру: патчи как токены, CLS, positional embeddings; чем ViT отличается от CNN.
3. **Swin Transformer** — ViT с оконным вниманием и иерархией (как у CNN), вычислительно эффективный.
4. **Self-supervised в CV** — contrastive (SimCLR/MoCo), non-contrastive (DINO/BYOL), masked image modeling (MAE).
5. **CLIP** — два энкодера в общем пространстве, симметричный InfoNCE, zero-shot классификация.
6. **Блиц** — закрепляем основные концепты.

> **Главная мысль:** картинка для трансформера — это **последовательность патчей-токенов**. Дальше работает всё то же, что мы уже знаем про attention.
"""
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
md(
    """
## 0. Recap и setup

Короткий мост из NLP-блока — что мы переносим в зрение:

| Знаем из NLP | Переносим в CV |
|---|---|
| Текст → последовательность токенов | Картинка → последовательность **патчей** |
| Self-attention между токенами | Self-attention между патчами |
| CLS-токен агрегирует последовательность (`BERT`) | CLS-токен агрегирует картинку (`ViT`) |
| Positional encoding (порядок слов) | Positional embeddings (позиция патча) |
| Self-supervised: MLM (предсказать замаскированное слово) | MAE (восстановить замаскированный патч) |

Картинку-пример возьмём один раз и переиспользуем во всех блоках.
"""
)

code(
    """
# В Colab при необходимости раскомментируйте:
# !pip install -q datasets transformers

import torch
from datasets import load_dataset

torch.manual_seed(0)

# Картинка-пример: кошки (та же, что в HF-туториалах). Используем её везде ниже.
dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]
image
"""
)

# ---------------------------------------------------------------------------
# Part I: ViT
# ---------------------------------------------------------------------------
md(
    """
## 1. Vision Transformer (ViT)

Начнём с **ванильного** ViT — именно он лежит в основе всех остальных архитектур этого семинара. Сначала запустим готовую модель (Outside-In: сперва «как пользоваться»), потом разберём, что у неё внутри.

ViT впервые предложили в статье [«An Image is Worth 16x16 Words»](https://arxiv.org/pdf/2010.11929) — само название намекает на главную идею: картинка для трансформера это просто последовательность патчей-«слов».

<img width="640" src=\""""
    + IMG_BASE
    + """/an_image_worth_1616_words.png" alt="An Image is Worth 16x16 Words — иллюстрация к идее ViT: картинка как последовательность патчей-токенов">
"""
)

code(
    """
from transformers import AutoImageProcessor, ViTForImageClassification

# Классический ViT, предобученный на ImageNet (patch 16x16, вход 224x224).
vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

inputs = vit_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = vit(**inputs).logits

# Модель предсказывает один из 1000 классов ImageNet.
predicted_label = logits.argmax(-1).item()
print(vit.config.id2label[predicted_label])
"""
)

md(
    """
### 1.1 Как устроен ViT

Трансформер умеет работать с **последовательностью векторов**. Чтобы скормить ему картинку, её надо превратить в такую последовательность:

1. **Патчификация.** Картинку `224x224` режут на непересекающиеся патчи `16x16`. Получается $\\left(\\frac{224}{16}\\right)^2 = 14 \\times 14 = 196$ патчей — это и есть «токены».
2. **Patch embedding.** Каждый патч (вектор из $16 \\times 16 \\times 3 = 768$ чисел) линейно проецируется в эмбеддинг размерности $d$. Технически это удобно сделать одной свёрткой с `kernel_size = stride = patch_size`.
3. **CLS-токен.** В начало последовательности добавляют обучаемый CLS-токен — ровно как в `BERT`. Его выход после энкодера идёт в классификатор.
4. **Positional embeddings.** К каждому токену прибавляют обучаемый вектор позиции — иначе трансформер не знает, где находился патч (attention сам по себе не учитывает порядок).
5. Дальше — обычный **Transformer Encoder** (multi-head self-attention + MLP), который мы уже разбирали.

Вся схема целиком:

<img width="760" src=\""""
    + IMG_BASE
    + """/vit_pipeline.png" alt="Схема ViT: картинка → патчи → patch embedding → +CLS +positional → Transformer Encoder ×L → выход CLS → MLP Head → класс">

Посмотрим на патчи руками.
"""
)

code(
    '''
import matplotlib.pyplot as plt


def patchify(img: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """Разбить изображение [C, H, W] на последовательность патчей [N, C, P, P].

    Args:
        img: Тензор изображения формы [C, H, W].
        patch_size: Размер стороны квадратного патча P.

    Returns:
        Тензор [N, C, P, P], где N = (H/P) * (W/P) — число патчей.
    """
    c, h, w = img.shape
    assert h % patch_size == 0 and w % patch_size == 0, "H и W должны делиться на patch_size"
    patches = (
        img.unfold(1, patch_size, patch_size)        # режем по высоте
           .unfold(2, patch_size, patch_size)        # режем по ширине -> [C, H/P, W/P, P, P]
           .permute(1, 2, 0, 3, 4)                   # -> [H/P, W/P, C, P, P]
           .reshape(-1, c, patch_size, patch_size)   # -> [N, C, P, P]
    )
    return patches


pixel_values = inputs["pixel_values"][0]              # [3, 224, 224]
patches = patchify(pixel_values, patch_size=16)
print("Картинка:", tuple(pixel_values.shape), "-> патчей:", patches.shape[0], "по", tuple(patches.shape[1:]))
'''
)

code(
    '''
# Визуализируем патчи в виде сетки 14x14 — это и есть "токены" для трансформера.
grid = pixel_values.shape[1] // 16
fig, axes = plt.subplots(grid, grid, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    p = patches[i].permute(1, 2, 0)
    p = (p - p.min()) / (p.max() - p.min() + 1e-8)    # денормализация только для показа
    ax.imshow(p)
    ax.axis("off")
plt.suptitle(f"{grid}x{grid} = {grid * grid} патчей — последовательность 'токенов' для ViT")
plt.tight_layout()
plt.show()
'''
)

code(
    '''
import torch.nn as nn

# Patch embedding = свёртка с kernel=stride=patch_size: каждый патч -> вектор размерности embed_dim.
embed_dim, patch_size = 768, 16
patch_embed = nn.Conv2d(in_channels=3, out_channels=embed_dim,
                        kernel_size=patch_size, stride=patch_size)

tokens = patch_embed(pixel_values.unsqueeze(0))       # [1, 768, 14, 14]
tokens = tokens.flatten(2).transpose(1, 2)            # [1, 196, 768]
print("Последовательность токенов:", tuple(tokens.shape), "= (batch, 196 патчей, 768)")
print("Дальше: + CLS-токен, + positional embeddings -> обычный Transformer Encoder")
'''
)

md(
    """
### 1.2 ViT vs CNN: inductive bias

У свёрточных сетей встроены сильные предположения о картинках (**inductive bias**): локальность (соседние пиксели связаны) и трансляционная инвариантность (кошка остаётся кошкой при сдвиге). ViT этих предположений почти **не** имеет — он смотрит на все патчи глобально с первого слоя и должен выучить структуру изображений «с нуля».

Плата за гибкость: **ViT жаден до данных**. На небольших датасетах CNN обычно выигрывает, но при предобучении на огромных корпусах (ImageNet-21k, JFT-300M) ViT догоняет и обгоняет свёртки. Поэтому ViT почти всегда используют как **предобученную** модель.
"""
)

md(
    """
#### ❓ **Вопрос**: почему ViT требует больше данных для обучения, чем CNN сопоставимого размера?

<details>

<summary><strong>Ответ</strong></summary>

У CNN в архитектуру «зашиты» полезные предположения о картинках — локальность и трансляционная инвариантность. Это бесплатный inductive bias, который не нужно выучивать из данных.</br>
У ViT таких предположений почти нет: глобальный self-attention с первого слоя должен сам выучить, что пиксели/патчи рядом связаны и что объект инвариантен к сдвигу. Чтобы выучить эти закономерности «с нуля», нужно много данных.</br>
Поэтому ViT обычно предобучают на очень больших датасетах, а потом дообучают (fine-tune) под конкретную задачу.

</details>
"""
)

# ---------------------------------------------------------------------------
# Part 2: Swin
# ---------------------------------------------------------------------------
md(
    """
## 2. Swin Transformer

У ванильного ViT self-attention **глобальный**: каждый патч смотрит на все остальные, поэтому сложность растёт как $O(n^2)$ от числа патчей. Для картинок высокого разрешения это дорого.

**Swin** (Shifted **WIN**dows, статья [«Swin Transformer: Hierarchical Vision Transformer using Shifted Windows»](https://arxiv.org/pdf/2103.14030)) чинит это двумя идеями:

1. **Window attention.** Attention считается только внутри небольших окон (например, $7 \\times 7$ патчей), а не по всей картинке. Сложность становится **линейной** по числу патчей.
2. **Shifted windows.** Если бы окна всегда стояли на одном месте, патчи из разных окон никогда бы не «общались». Поэтому на следующем слое окна **сдвигают** — так информация перетекает между соседними окнами.

<img width="640" src=\""""
    + IMG_BASE
    + """/swin_window_attention.png" alt="Global attention (ViT, O(n^2)) против window attention (Swin, O(n)): слева патч смотрит на все патчи, справа — только внутри своего окна">

А вот как работает сдвиг окон между соседними слоями — границы окон смещаются, и патчи, которые раньше были в разных окнах, попадают в одно:

<img width="640" src=\""""
    + IMG_BASE
    + """/swin_shifted.png" alt="Shifted windows в Swin: на следующем слое сетка окон сдвигается, поэтому информация перетекает между соседними окнами">

Плюс **иерархия**: как в CNN, Swin постепенно уменьшает разрешение и увеличивает число каналов (`patch merging`), строя многоуровневые признаки. Это делает Swin удобным backbone для детекции и сегментации.

<img width="640" src=\""""
    + IMG_BASE
    + """/swin_patches.png" alt="Иерархия патчей в Swin: мелкие патчи на первых слоях постепенно объединяются (patch merging) в более крупные, формируя многоуровневое представление как в CNN">

Запустим Swin на той же картинке.
"""
)

code(
    """
from transformers import SwinForImageClassification

swin_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
swin = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

swin_inputs = swin_processor(image, return_tensors="pt")

with torch.no_grad():
    swin_logits = swin(**swin_inputs).logits

print(swin.config.id2label[swin_logits.argmax(-1).item()])
"""
)

md(
    """
Разбор исходников `transformers` (для самостоятельного чтения):

- Window attention: [`modeling_swin.py#L822`](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/swin/modeling_swin.py#L822)
- Сборка блоков: [`modeling_swin.py#L990`](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/swin/modeling_swin.py#L990)
"""
)

# ---------------------------------------------------------------------------
# Part 3: Self-supervised in CV
# ---------------------------------------------------------------------------
md(
    """
## 3. Self-supervised learning в компьютерном зрении

Разметка картинок дорогая, а неразмеченных изображений — бесконечно много. **Self-supervised learning (SSL)** придумывает задачу-предлог (pretext task) прямо из самих данных, без человеческой разметки — ровно как MLM в `BERT`. Получаем сильные представления, которые потом дообучаем под конкретную задачу.

Три больших семейства SSL в зрении:

| Семейство | Идея | Примеры | Нужны негативы? |
|---|---|---|---|
| **Contrastive** | сблизить две аугментации одной картинки (позитивы), оттолкнуть разные картинки (негативы) | SimCLR, MoCo | да |
| **Non-contrastive** | сблизить два вида одной картинки без явных негативов (хитрости против коллапса) | BYOL, DINO | нет |
| **Masked image modeling** | замаскировать часть патчей и восстановить их (аналог MLM) | MAE, BEiT | нет |

Ключевая механика contrastive-подхода — **позитивная пара**: две случайные аугментации одной и той же картинки. Модель учат выдавать им близкие эмбеддинги. Посмотрим, как выглядит такая пара.
"""
)

code(
    '''
import torchvision.transforms as T

# SimCLR-аугментации: из одной картинки делаем две случайные "вьюхи" = позитивная пара.
simclr_aug = T.Compose([
    T.RandomResizedCrop(160, scale=(0.4, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.RandomGrayscale(p=0.2),
])

view1, view2 = simclr_aug(image), simclr_aug(image)

fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].imshow(view1); ax[0].set_title("view 1"); ax[0].axis("off")
ax[1].imshow(view2); ax[1].set_title("view 2"); ax[1].axis("off")
plt.suptitle("Позитивная пара: одна картинка -> две аугментации (их эмбеддинги сближаем)")
plt.tight_layout()
plt.show()
'''
)

md(
    """
#### ❓ **Вопрос**: в contrastive learning что выступает позитивной парой, а что — негативной? Зачем вообще нужны негативы?

<details>

<summary><strong>Ответ</strong></summary>

Позитивная пара — две разные аугментации **одной и той же** картинки; их эмбеддинги мы сближаем.</br>
Негативы — это другие картинки в батче; их эмбеддинги мы отталкиваем.</br>
Без негативов есть риск **коллапса**: модель может выдавать один и тот же вектор на всё подряд — тогда позитивы «совпадают» идеально, но представления бесполезны. Негативы заставляют пространство быть различающим. (Non-contrastive методы вроде BYOL/DINO борются с коллапсом другими трюками — stop-gradient, momentum-энкодер, центрирование.)

</details>
"""
)

md(
    """
#### ❓ **Вопрос**: чем self-supervised обучение отличается от unsupervised и от supervised?

<details>

<summary><strong>Ответ</strong></summary>

* **supervised** — есть человеческая разметка (метки классов, переводы и т.п.). Размеченных данных обычно на порядки меньше, чем сырых.</br>
* **unsupervised** — целевых меток нет вообще, цель — найти структуру в данных (например, кластеризация).</br>
* **self-supervised** — меток от человека нет, но задачу-цель мы конструируем из самих данных (предсказать замаскированный патч/слово, сблизить аугментации). Формально метки «генерируются автоматически», поэтому это частный случай unsupervised, но с supervised-подобной лоссовой задачей.</br>

Типичный рецепт: self-supervised **предобучение** на куче неразмеченных данных, затем **дообучение** на маленьком размеченном датасете под конкретную задачу.

</details>
"""
)

# ---------------------------------------------------------------------------
# Part 4: CLIP
# ---------------------------------------------------------------------------
md(
    """
## 4. CLIP — Contrastive Language–Image Pre-training

**CLIP** — это contrastive learning, но между **двумя модальностями**: текстом и картинкой. У модели два энкодера (image encoder — ViT, text encoder — трансформер), которые проецируют картинку и текст в **общее векторное пространство**. Обучают на ~400M пар «картинка ↔ её подпись из интернета»: эмбеддинги совпадающих пар сближают, несовпадающих — отталкивают.

<img width="640" src=\""""
    + IMG_BASE
    + """/clip_dual_encoder.png" alt="Обучение CLIP: image encoder и text encoder дают эмбеддинги, считается матрица сходств N×N; диагональ (совпадающие пары) сближаем, остальное отталкиваем">

Сначала — как этим пользоваться: **zero-shot классификация** без единого примера обучения под наши классы.
"""
)

code(
    """
from transformers import CLIPProcessor, CLIPModel

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Произвольные классы, оформленные как текстовые описания ("prompt").
classes = ["a photo of a cat", "a photo of a dog", "a photo of two cats", "a photo of a car"]

clip_inputs = clip_processor(text=classes, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = clip(**clip_inputs)

# logits_per_image: близость картинки к каждому тексту; softmax -> "вероятности классов".
probs = outputs.logits_per_image.softmax(dim=1)[0]
for cls, p in sorted(zip(classes, probs.tolist()), key=lambda x: -x[1]):
    print(f"{p:.3f}  {cls}")
"""
)

md(
    """
### 4.1 Zero-shot рецепт и prompt engineering

CLIP **не обучался** на наши классы — мы просто описали каждый класс текстом и спросили, какой текст ближе к картинке. Это и есть **zero-shot**: задача, на которую модель не училась напрямую.

Несколько практических деталей:

- Класс оформляют как **фразу**, а не одно слово: `"a photo of a {class}"` работает лучше, чем просто `"cat"` — так распределение текста ближе к подписям, на которых училась модель.
- Можно усреднять несколько шаблонов (`"a photo of a {}"`, `"a blurry photo of a {}"`, ...) — это **prompt ensembling**, заметно поднимает точность.
- На вход и текст, и картинка идут как последовательности; **единый эмбеддинг** берётся со специального агрегирующего токена (как CLS в `BERT`) и проецируется в общее пространство.

Посмотрим на сырые эмбеддинги и косинусную близость напрямую — это база и для retrieval, и для CLIP-score.
"""
)

code(
    """
import torch.nn.functional as F

texts = ["a cat", "a dog", "a sofa", "two cats sleeping"]
ti = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    out = clip(**ti)
    img_emb = out.image_embeds   # [1, d] — картинка в общем пространстве
    txt_emb = out.text_embeds    # [len(texts), d] — тексты там же

# L2-нормировка -> скалярное произведение = косинусная близость.
img_emb = F.normalize(img_emb, dim=-1)
txt_emb = F.normalize(txt_emb, dim=-1)

sims = (img_emb @ txt_emb.T)[0]
print("Косинусная близость картинки к текстам:")
for t, s in sorted(zip(texts, sims.tolist()), key=lambda x: -x[1]):
    print(f"  cos = {s:+.3f}   {t}")
"""
)

md(
    """
### 4.2 Лосс CLIP: симметричный InfoNCE

**InfoNCE** (*Info Noise-Contrastive Estimation*) — это базовый contrastive-лосс: «найди среди многих кандидатов единственный правильный». Именно его оптимизирует CLIP.

Берём батч из $N$ пар $(I_i, T_i)$ — картинка и её подпись. Энкодеры дают эмбеддинги, которые **L2-нормируем**; тогда близость любой пары — это косинус $s_{ij} = I_i^{\\top} T_j$. Складываем все близости в матрицу $N \\times N$: на диагонали стоят совпадающие пары (позитивы), вне диагонали — негативы (картинка с чужими подписями из того же батча).

Для каждой картинки $i$ правильный текст среди всех $N$ — только её собственный $T_i$. Это задача **классификации**, то есть softmax-cross-entropy по строке матрицы:

$$\\mathcal{L}_{\\text{img}\\to\\text{txt}} = -\\frac{1}{N}\\sum_{i=1}^{N} \\log \\frac{\\exp(s_{ii}/\\tau)}{\\sum_{j=1}^{N}\\exp(s_{ij}/\\tau)}$$

Симметрично — то же самое для текстов (для каждого текста ищем его картинку среди всех), то есть softmax по столбцам:

$$\\mathcal{L}_{\\text{txt}\\to\\text{img}} = -\\frac{1}{N}\\sum_{i=1}^{N} \\log \\frac{\\exp(s_{ii}/\\tau)}{\\sum_{j=1}^{N}\\exp(s_{ji}/\\tau)}$$

Итоговый лосс — среднее двух направлений:

$$\\mathcal{L}_{\\text{CLIP}} = \\tfrac{1}{2}\\left(\\mathcal{L}_{\\text{img}\\to\\text{txt}} + \\mathcal{L}_{\\text{txt}\\to\\text{img}}\\right)$$

**Интуиция:** числитель тянет вверх совпадающую пару (диагональ), знаменатель отталкивает все остальные (негативы). Это ровно cross-entropy, где «правильный класс» строки $i$ — столбец $i$.

Здесь $\\tau$ — **температура** (масштаб логитов). В коде ниже мы намеренно берём $\\tau = 1$ (т.е. вообще не масштабируем), чтобы своими глазами увидеть проблему — а *как* выбрать и реализовать $\\tau$, вы разберётесь в домашке `clip`.

<img width="560" src=\""""
    + IMG_BASE
    + """/infonce_matrix.png" alt="InfoNCE как классификация по матрице сходств N×N: цель — диагональ; softmax по строкам (text→image) и по столбцам (image→text), лосс симметричный">

Соберём лосс руками на игрушечном батче.
"""
)

code(
    '''
import torch.nn as nn


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """Cross-entropy, где правильный класс для строки i — это столбец i (совпадающая пара).

    Args:
        logits: Матрица сходства [N, N].

    Returns:
        Скаляр — средняя cross-entropy по строкам.
    """
    targets = torch.arange(len(logits), device=logits.device)
    return nn.functional.cross_entropy(logits, targets)


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    """Симметричный InfoNCE: cross-entropy и по строкам, и по столбцам."""
    return (contrastive_loss(similarity) + contrastive_loss(similarity.t())) / 2.0


# Игрушечный батч из 8 пар (картинка_i <-> текст_i):
image_embeds = torch.rand(8, 512)
text_embeds = torch.rand(8, 512)

# Шаг 1: L2-нормировка эмбеддингов -> косинусная близость.
image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

# Шаг 2: матрица сходства всех пар. ВНИМАНИЕ: значения лежат в [-1, 1].
logits_per_text = text_embeds @ image_embeds.t()
print("Диапазон логитов:", round(logits_per_text.min().item(), 3),
      "..", round(logits_per_text.max().item(), 3))

loss = clip_loss(logits_per_text)
print("CLIP loss:", round(loss.item(), 4))
'''
)

md(
    """
#### ❓ **Вопрос (мотивация к домашке)**: мы кормим в softmax логиты из диапазона $[-1, 1]$. Что это значит для итогового распределения вероятностей и насколько «уверенным» сможет стать обучение?

<details>

<summary><strong>Подсказка</strong></summary>

Softmax от логитов в $[-1, 1]$ даёт почти **равномерное** распределение: даже у идеально совпадающей пары вероятность не может приблизиться к 1, а градиент остаётся слабым — модель почти не может стать «уверенной».</br>
Лечится это **масштабированием** логитов перед softmax. А вот *какой* множитель выбрать, должен ли он быть **обучаемым** или фиксированным, почему его удобно хранить и применять определённым образом — вы разберётесь и внедрите **сами в домашке `clip`**. Это ключевая часть задания, поэтому здесь намеренно оставляем вопрос открытым.

</details>
"""
)

# ---------------------------------------------------------------------------
# Блиц
# ---------------------------------------------------------------------------
md("# Блиц")

md(
    """
### Vision Transformers
"""
)

md(
    """
#### ❓ **Вопрос**: чем `Swin` отличается от `ViT`?

<details>

<summary><strong>Ответ</strong></summary>

| Характеристика   | ViT                   | Swin                                 |
| ---------------- | --------------------- | ------------------------------------ |
| Область внимания | Глобальная            | Локальные окна со сдвигом            |
| Сложность attention | $O(n^2)$           | $O(n)$ (линейная по числу патчей)   |
| Иерархия         | Нет                   | Есть (как у CNN, через patch merging)|
| Inductive bias   | Мало                  | Больше (локальность, как у CNN)     |

</details>
"""
)

md(
    """
#### ❓ **Вопрос**: что такое Shifted Window из названия `Swin`? Для чего оно используется?

<details>

<summary><strong>Ответ</strong></summary>

Чтобы сделать attention дешёвым, `Swin` считает его только внутри небольших окон. Но при фиксированных окнах патчи из разных окон никогда бы не «общались».</br>
Поэтому на следующем слое окна **сдвигают** (shifted windows): границы окон смещаются, и информация перетекает между соседними окнами. Так сохраняется и линейная сложность, и связь между удалёнными областями.

</details>
"""
)

md(
    """
### Self-supervised & Contrastive learning
"""
)

md(
    """
#### ❓ **Вопрос**: что такое zero-shot learning? Приведите пример модели и задачи. Почему это работает?

<details>

<summary><strong>Ответ</strong></summary>

Zero-shot learning — модель решает задачу, на которой её **напрямую не обучали**.</br>
Пример: `CLIP` и классификация. CLIP не обучался классифицировать наши классы, но мы описываем каждый класс текстом и берём ближайший к картинке — задача классификации сводится к задаче «найди ближайший текст».</br>
Работает потому, что CLIP обучался на **более общую** задачу (сопоставление картинок и текстов), к которой классификация сводится как частный случай.

</details>
"""
)

md(
    """
#### ❓ **Вопрос**: что такое мультимодальное обучение? Какие модальности бывают?

<details>

<summary><strong>Ответ</strong></summary>

Модальность — это тип воспринимаемой информации. Мультимодальная модель работает сразу с несколькими типами.</br>
`CLIP` мультимодален: текст (первая модальность) + изображение (вторая).</br>
Другие модальности: видео, аудио, облака точек, карты глубины, тепловые карты.</br>

`CLIP` на максималках (6 модальностей в одном пространстве): [ImageBind](https://github.com/facebookresearch/ImageBind).

</details>
"""
)

md(
    """
#### ❓ **Вопрос**: какая лосс-функция оптимизируется при обучении `CLIP`?

<details>

<summary><strong>Ответ</strong></summary>

Симметричный **InfoNCE** (contrastive loss). В батче из $N$ пар максимизируется близость совпадающих пар картинка-текст и минимизируется для всех негативных пар.</br>
Перед вычислением близости эмбеддинги **L2-нормируются** (близость = косинус). Получается матрица $N \\times N$, и cross-entropy считается симметрично — по строкам и по столбцам.</br>
Логиты перед softmax дополнительно **масштабируются** — зачем именно и как это правильно реализовать, разбирается в домашке `clip`.

</details>
"""
)

md(
    """
#### ❓ **Вопрос**: какие ещё задачи, кроме zero-shot классификации, можно решать через CLIP-эмбеддинги?

<details>

<summary><strong>Ответ</strong></summary>

* поиск и ранжирование картинок по текстовому запросу (поиск в векторном пространстве);
* поиск похожих картинок по картинке;
* **CLIP-score** как метрика: насколько сгенерированная картинка соответствует текстовому запросу (используется при оценке text-to-image генерации);
* guidance для диффузионных моделей (направлять генерацию текстом).

</details>
"""
)

md(
    """
#### ❓ **Вопрос**: `CLIP` получает на вход последовательность (и текст, и картинку), но contrastive learning работает с одним вектором-эмбеддингом. Как этот вектор выбирается?

<details>

<summary><strong>Ответ</strong></summary>

И для текста, и для картинки используется специальный агрегирующий токен (CLS-подобный), который «стягивает» в себя информацию со всей последовательности. Его выход проецируется в общее пространство.</br>
Аналогично делали для классификации в `BERT`.

</details>
"""
)

# ---------------------------------------------------------------------------
# Дополнительные материалы
# ---------------------------------------------------------------------------
md(
    """
# Дополнительные материалы

**Архитектуры:**
- ViT — [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Swin — [Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

**Self-supervised:**
- SimCLR — [A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709)
- MoCo — [Momentum Contrast](https://arxiv.org/abs/1911.05722)
- BYOL — [Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733)
- DINO — [Emerging Properties in Self-Supervised ViT](https://arxiv.org/abs/2104.14294)
- MAE — [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

**Мультимодальность:**
- CLIP — [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- ImageBind — [One Embedding Space To Bind Them All](https://github.com/facebookresearch/ImageBind)
- LLaVA — [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)

**Курсы:** [CS231n](https://cs231n.github.io/) (CV), [d2l.ai](https://d2l.ai/) (раздел про attention и ViT).
"""
)


# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------
NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "13_seminar_vit_clip.ipynb")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, ensure_ascii=False, indent=1)
    f.write("\n")

print(f"Wrote {OUT} ({len(CELLS)} cells)")
