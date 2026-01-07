# Материалы курса Глубинное обучение

* Прогон курса в [2024 году](https://github.com/fintech-dl-hse/course/tree/2024)
* Прогон курса в [2023 году](https://github.com/fintech-dl-hse/course/tree/2023)
* Прогон курса в [2022 году](https://github.com/fintech-dl-hse/course/tree/2022)
* [Разборы семинаров 2022 года](https://youtube.com/playlist?list=PLCNrwCOlxMPxUnJNtthxVdL3eYAj0Lp1Q)

### Планируется

* ~15 лекций и семинаров
* ~12 домашек + 3 бонусных домашки

Материалы будут выкладываться в течение курса.

**Обратная связь.** пожалуйста, заполняйте формы обратной связи. Желательно заполнять открытые вопросы --- в них можете высказать все, что угодно)

**Сложность курса.** Понятно, что на курсе есть люди с разными бэкграундами. Если вам что-то не понятно, не стесняйтесь задавать вопрос.

## Как задавать вопросы?

* не стесняться, не бывает глупых вопросов
* в общий чатик --- так есть вероятность, что вам может помочь кто-то из одногруппников быстрее, чем преподаватель и на преподавателей меньше нагрузка
* на лекциях или семинарах
* в личку


## Как построены семинары?

* Работа над ошибками по обратной связи. Если че-то не нравится на семах, пишите в формочки обратной связи, будем думать
* Повторение, что прошли на прошлом семе
* Новый материал
* Блиц по новому материалу
* Обзор домашки, если она есть

## Домашки (TODO обновить дедлайны и список ДЗ)

| Домашка                  | Количество баллов | Дедлайн                           |
| ------------------------ | ----------------- | --------------------------------- |
| mlp                      | 1                 |  TODO  |
| weight-init              | 1                 |  TODO  |
| activations              | 1                 |  TODO  |
| optimization             | 2                 |  TODO  |
| batchnorm                | 1                 |  TODO  |
| dropout                  | 1                 |  TODO  |
| pytorch-basics           | 2                 |  TODO  |
| vae                      | 2                 |  TODO  |
| diffusion                | 2                 |  TODO  |
| tokenization             | 2                 |  TODO  |
| rnn-attention            | 2                 |  TODO  |
| transformer-attention    | 2                 |  TODO  |
| llm-agent                | 2                 |  TODO  |
| clip                     | 1                 |  TODO  |



Максимум можно получить 16 баллов за все домашки.

## Бонусные домашки (TODO обновить дедлайны и список ДЗ)

| Домашка                  | Количество баллов | Дедлайн                           |
| ------------------------ | ----------------- | --------------------------------- |
| letters                  | 4                 |      15.06 09:00 (конец курса)    |
| multimodal-llm           | 4                 |      15.06 09:00 (конец курса)    |

Бонусные домашки посложнее, чем обычные домашки, но за них можно получить много баллов. С бонусными домашками можно получить больше 10 баллов за ДЗ.

**Списывание (!):** За списывание зануляются все работы. Если используете код из открытых источников, пожалуйста указывайте ссылки.

**Сдача:** Домашки будут сдаваться в github classroom.

В домашках настроено автоматическое оценивание. Временем сдачи домашки будет считаться
время, когда прошли пайплайны (то есть не время коммита, а время пуша + время на прогон тестов).
За попытки хакнуть, поломать или обмануть автогрейдер балл за домашку зануляется.

За неоптимальные решения или за неправильные решения баллы могут снижаться.

**Штрафы:** За просрочку за каждый день будет сниматься по `10%` от оценки, но суммарно штраф не может быть более `30%`. Жестких дедлайнов нет. Если сдаешь через 3 дня домашку, штраф `30%`. Если сдаешь через месяц, штраф тоже `30%`

## Формула

Итоговая формула - взвешенная сумма оценки за домашки и за экзамен.

$$ O_{hse} = 0.8 \cdot O_{hw} + 0.1 \cdot O_{exam3} + 0.1 \cdot O_{exam4} $$

$ O_{exam3} $ и $ O_{exam4} $ - оценки за экзамен соответственно 3 и 4 модуля

### Экзамен

**Допуск:** Допуском к экзамену будет прохождение тестов по всем пройденным темам с автоматической проверкой.

**Список вопросов к экзамену** TODO


## Примерный план

12 января занятий не будет. Первое занятие состоится 19 января.

| Дата  | Лекция                                                           | Домашка                          |
|-------|------------------------------------------------------------------|----------------------------------|
| 19.01 | Введение в глубинное обучение. Обучение нейросетей. Алгоритм обратного распространения ошибки. | mlp |
| 26.01 | Функции активации. Задачи и функции потерь. Инициализация весов. Lottery Ticket Hypothesis | weight-init, activations         |
| 02.02 | Оптимизация. SGD, Adam, Muon. Регуляризация: dropout, weight decay | optimization, dropout            |
| 09.02 | Сверточные сети.                                                 | pytorch-basics, batchnorm        |
| 16.02 | Задачи Computer Vision.                                          | letters (бонусная)               |
| 23.02 | -                                                                |                                  |
| 02.03 | Генеративные модели: Авторегрессионные, GAN.                     |                                  |
| 09.03 | Генеративные модели: VAE, Diffusion.                             | vae, diffusion                   |
| 16.03 | -                                                                |                                  |
| 23.03 | -                                                                |                                  |
| 30.03 | _экзамен_                                                        |                                  |
| 06.04 | NLP, Word2vec. Tokenization: BPE, WordPiece, SentencePiece       | tokenization                     |
| 13.04 | Рекуррентные сети. Attention, Transformers. Positional Encoding, RoPE, YaRN. | transformer-attention, rnn-attention |
| 20.04 | Pretrained transformers in NLP. Self-Supervised Learning. Large Language Models.  |                 |
| 27.04 | -                                                                |                                  |
| 04.05 | Scaling Laws. In context learning. Test time scaling (thinking models). PEFT. | efficiency          |
| 11.05 | -                                                                |                                  |
| 18.05 | Function Calling. Agentic patterns. Observability. MCP. RAG.     | llm-agent (+mcp)                 |
| 01.06 | Vision Transformers. Self-supervised, contrastive learning.      | clip                             |
| 08.06 | Мультимодальные модели                                           | multimodal-llm (бонусная)        |
| 22.06 | -                                                                |                                  |
| 15.06 | -                                                                |                                  |
| 17.06 | _экзамен_                                                        |                                  |

**Дополнительные темы:**
* Эволюционные алгоритмы и LLM - AlphaEvolve
* Методы интерпретируемости глубоких нейросетевых моделей - SAE, Activation Patching
* Устройство GPU. Низкоуровневое программирование на Triton
* Эффективность LLM (квантизация, спекулятивный декодинг, дистилляция, прунинг)
* GNN - графовые нейросети


## Литература

* *Глубокое обучение. Погружение в мир нейронных сетей.* С. Николенко, А. Кадурин, Е. Архангельская.

* *Глубокое обучение.* Я. Гудфеллоу, Й. Бенджио, А. Курвилль.
Есть английская версия: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/).

* *Understanding deep learning.* S. Prince.
[https://udlbook.github.io/udlbook/](https://udlbook.github.io/udlbook/).

* *Dive into Deep Learning.* A. Zhang, Z. Lipton, M. Li, A.
Smola.
[https://d2l.ai/](https://d2l.ai/).


## Полезные материалы

* [Материалы с курса бакалавриата](https://github.com/aosokin/dl_cshse_ami/tree/master/2021-fall)
* [Курс ШАДа](https://github.com/yandexdataschool/Practical_DL)
* [Курс Стендфорда про CV with DL](https://cs231n.github.io/)
* [Курс Стендфорда про NLP with DL](http://web.stanford.edu/class/cs224n/) --- один из лучших курсов про NLP
* [DLSchool](https://www.dlschool.org/) --- много классных домашек и проект в конце курса, там есть и про NLP, и про CV, на некоторых прогонах затрагивали и работу со звуком (на ютубе можно найти записи прогонов прошлых лет, на степики старые прогоны курса тоже можно найти)
* [Вводный курс Семена Козлова](http://dlcourse.ai/)
* [Lena Voita](https://lena-voita.github.io) --- классный блог и курс по NLP
* [Jay Alammar](https://jalammar.github.io/), [Sebastian Ruder](https://ruder.io/) --- еще популярные блоги про NLP
* [distill.pub](https://distill.pub/) --- журнал с красивыми визуализациями
* [paperswithcode](https://paperswithcode.com/) --- сравнение разных архетектур/задач
* [How To Read Papers](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf)
* [Annotated deep learning paper implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations)

## Годные телеграм-каналы

* [Сиолошная](https://t.me/seeallochnaya) - канал Игоря Котенкова, у него и [youtube-канал годный](https://www.youtube.com/@stalkermustang/featured)
* [gonzo-обзоры ML статей](https://t.me/gonzo_ML) - текстовые разборы статей
* [AIRI Community](https://t.me/c/2345326588/1) - AIRI
* [GigaDev — разработка GigaChat](https://t.me/gigadev_channel) - Команда разработки GigaChat
* [Yandex for ML](https://t.me/yandexforml) - обратите внимание [на закрепленный пост](https://t.me/yandexforml/195)


## Что еще?

**Гуглите:** эти и многие другие материалы легко находятся, если вы пытаетесь разобраться в какой-то теме.

**Читайте документацию:** в [Pytorch Docs](https://pytorch.org/docs/stable/index.html), [Pytorch Tutorials](https://pytorch.org/tutorials/) можно найти и описание методов, и формулы, и ссылки на статьи.

**Читайте статьи:** большинство концептов, которые мы проходим в этом курсе, были опубликованы в статьях, которые доступны на [arxiv](https://arxiv.org/). Где следить за самыми современными методами: конференции NeurIPS, ICML, ICLR, CVPR, ACL, блоги крупных компаний [Google AI](https://ai.googleblog.com/), [DeepMind](https://deepmind.com/blog), [OpenAI](https://openai.com/blog/), [Meta AI](https://ai.facebook.com/blog/).
