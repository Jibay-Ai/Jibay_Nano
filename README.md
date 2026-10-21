---
license: apache-2.0
language:
- fa
tags:
- php
- 8K Params
- Jibay Nano
- BPE
- Transformer
- Emmbding
- Decoder
- MTP
pipeline_tag: text-generation
---

![Jibay Nano](https://huggingface.co/JibayAi/Jibay_nano/resolve/main/image_jibaynano.jpg)

# Jibay Nano

**A real Transformer-based neural network implemented in pure PHP**
**یک شبکهٔ عصبی واقعی مبتنی بر ترنسفورمر، پیاده‌سازی‌شده با PHP خالص**

Developed by the **JibayAI** team · وب‌سایت: [jibay.ir](https://jibay.ir)

---

# English

## 1. Overview

**Jibay Nano** is a real, working, decoder-only Transformer language model written **entirely in PHP**, with no external machine-learning libraries, no Python, no TensorFlow, no PyTorch, and no GPU. Everything — tokenizer training, forward pass, backward pass, gradient computation, the Adam optimizer, weight serialization, and autoregressive text generation — is implemented from scratch in PHP.

This is not a wrapper around a hosted API, and it is not a lookup table of canned answers. It is a genuine small neural network that learns from data via backpropagation and produces output by sampling from a probability distribution over its vocabulary.

The project exists to prove a point: **you can build a real AI architecture in PHP**. PHP is not a language designed for numerical computing, and it is not a suitable platform for large-scale machine learning. But "not suitable" is not the same as "impossible". Jibay Nano demonstrates that the architecture, the mathematics, and the training loop can all live in a single PHP file and run on an ordinary web host.

## 2. Model Specifications

| Property | Value |
|---|---|
| Model name | Jibay Nano |
| Architecture class | `JibayNanoForCausalLM` |
| Model type | Decoder-only causal language model (GPT-style) |
| Parameters | **8,394** trainable weights |
| Layers (`n_layers`) | **2** Transformer blocks |
| Hidden dimension (`d_model`) | **8** |
| Attention heads (`n_heads`) | **2** (head dim = 4) |
| Feed-forward dimension (`ff_dim`) | **16** |
| Activation | GELU |
| Normalization | Pre-LayerNorm |
| Attention type | Causal multi-head self-attention |
| Vocabulary size | **257** tokens |
| BPE merges | **147** |
| Context length (`max_position_embeddings`) | **68** tokens |
| Max output tokens (`generation_limit`) | **75** tokens |
| Optimizer | Adam (β₁ = 0.9, β₂ = 0.999, ε = 1e-8) |
| Base learning rate | 0.001 (with warm-up over first 5 steps) |
| Batch size | 8 |
| Training examples | **210** Persian question–answer pairs |
| Training epochs | 10 |
| Training steps | 270 |
| Training time | ~776 seconds (CPU, single process) |
| Final training loss | 5.709 |
| Extra head | Multi-Token Prediction (MTP) with rejection sampling |
| Weight format | Binary float32 with magic header + architecture SHA-256 |

## 3. Architecture

Jibay Nano follows the standard **decoder-only Transformer** design used by GPT-family models, scaled down to the smallest possible size that still works correctly.

**Pipeline (forward pass):**

1. **Token embedding** — each input token id is mapped to an 8-dimensional vector.
2. **Positional embedding** — a learned position vector is added to each token.
3. **Transformer block × 2**, each containing:
   - Pre-LayerNorm → causal multi-head self-attention (2 heads) → residual connection
   - Pre-LayerNorm → feed-forward network (8 → 16 → 8) with GELU → residual connection
4. **Final LayerNorm**
5. **LM head** — linear projection from 8 → 257 logits.
6. **MTP head** — a second linear projection used by the Multi-Token Prediction auxiliary objective.

**Key design choices:**

- **Causal masking** — position *t* can only attend to positions ≤ *t*, which enforces autoregressive generation.
- **Pre-LayerNorm** — normalization happens before each sub-layer, which stabilizes training for small models.
- **GELU activation** — smoother than ReLU and standard in modern Transformers.
- **Weight tying is NOT used** — input embedding and output projection are separate matrices, which is simpler to implement and debug.

## 4. Tokenizer

The tokenizer is a **Byte-Pair Encoding (BPE)** model trained from scratch on the training corpus:

- Special tokens: `<|pad|>` (0), `<|bos|>` (1), `<|eos|>` (2), `<|unk|>` (3)
- Unicode-aware pre-tokenizer that groups letters, digits, and underscores into word pieces
- Whitespace, tabs, and newlines are mapped to the markers `Ġ`, `ĉ`, and `↵` respectively, so that spaces can be part of BPE merges
- **257** final vocabulary entries, produced by **147** merge operations
- Fully supports Persian characters, ZWNJ (`‌`), Persian digits, and Latin letters

The tokenizer is trained automatically the first time the model is built and stored in `vocab.json`, `merges.txt`, `tokenizer_config.json`, and `tokenizer.json`.

## 5. How It Works

### 5.1 Training

1. `dataset.json` is loaded (210 Persian Q&A examples).
2. The BPE tokenizer is trained on the corpus.
3. The model architecture is derived automatically from the dataset statistics (context length from the 90th percentile sequence length, generation limit from the 90th percentile answer length, model size from the number of examples).
4. For each batch, the forward pass computes logits; the backward pass computes gradients **analytically** for every weight (attention, LayerNorm, GELU, linear layers, embeddings).
5. Gradients are clipped, then applied with Adam.
6. Both the main next-token loss and an **MTP (Multi-Token Prediction)** auxiliary loss are optimized jointly — the model additionally learns to predict the token *two* steps ahead, which is used at inference for speculative decoding.
7. Checkpoints are saved after every batch, so training can resume after a crash or timeout.
8. Early stopping monitors validation loss with patience of 3 epochs.

### 5.2 Inference

1. The prompt is tokenized: `User: <question>\nAssistant:` with a `<|bos|>` prefix.
2. The context window is limited to the last 68 tokens.
3. For each generated token:
   - The forward pass produces main logits and MTP logits.
   - **Temperature**, **top-k**, and **top-p** sampling are applied.
   - The MTP head proposes a draft token for position *t+2*.
   - A **rejection sampling** step compares the draft distribution against the target distribution and accepts or rejects the draft. This is a speculative-decoding technique that, in large models, speeds up generation without changing the output distribution.
4. Special tokens (`<|bos|>`, `<|pad|>`, `<|unk|>`, and `<|eos|>` during the first two steps) are masked out.
5. Generation stops at `<|eos|>` or when the 75-token limit is reached.

## 6. Project Files

| File | Purpose |
|---|---|
| `index.php` | The entire model: tokenizer, training, inference, HTTP interface |
| `dataset.json` | Training data (210 Persian Q&A pairs, open) |
| `config.json` | Model configuration + training metadata |
| `model_config.json` | Mirror of `config.json` |
| `vocab.json` | BPE vocabulary (token → id) |
| `merges.txt` | BPE merge rules |
| `tokenizer_config.json` | Tokenizer metadata |
| `tokenizer.json` | Full tokenizer in Hugging Face-compatible layout |
| `model_weights.bin` | Trained weights (binary float32 + header) |
| `training_state.json` | Human-readable training progress |
| `training_checkpoint.tmp` | In-progress training checkpoint (temporary) |
| `training.lock` | File lock to prevent concurrent training |

## 7. Usage

Run `index.php` in any PHP environment (PHP 8.1+ recommended, no extensions required).

- **Model report:** open `index.php` with no parameters.
- **Ask a question:** `index.php?t=سلام`
- **Disable MTP:** `index.php?t=سلام&mtp=0`
- **Adjust sampling:** `index.php?t=سلام&temperature=0.7&top_k=20&top_p=0.85`
- **Continue training:** `index.php?train=1`

If `config.json` and `model_weights.bin` are present and consistent, the model loads without retraining. If the dataset changes, the existing weights are invalidated automatically and the model is rebuilt from scratch.

## 8. What This Model Is — and What It Is Not

**It is:**
- A real Transformer neural network with real learned weights.
- A real backpropagation implementation with real gradient descent.
- A working proof that the architecture of a modern language model can be built and trained in PHP.
- An open architecture for research, education, and experimentation.

**It is not:**
- A knowledgeable assistant. Do not expect it to answer questions about the world, science, history, or current events.
- A replacement for GPT, LLaMA, Claude, or any large language model.
- A model trained on a large or diverse corpus. It has 210 examples and 8,394 parameters.

The final training loss is **5.71**, and the theoretical loss of a perfectly random model over a 257-token vocabulary is **ln(257) ≈ 5.55**. This means the model has learned *some* structure — it is measurably better than guessing — but its capacity is far too small to be useful as a general-purpose assistant. **That is expected and intended.**

## 9. Why PHP, and Why Publish It

PHP is not a language designed for numerical computing. It is not optimized for matrix operations, it has no built-in tensor type, and its runtime is far slower than C, Rust, or CUDA. Every serious machine-learning library in the world is written in another language.

**We built Jibay Nano anyway — to prove that we can.**

We are publishing this architecture openly because we believe the value is not in the weights (which are tiny), but in the **design**: the fact that a complete GPT-style model — tokenizer, transformer, backpropagation, Adam, MTP, rejection sampling, weight serialization, checkpointing — can be written in a single PHP file and understood by reading it.

We invite the community to:

- **Improve it** — larger `d_model`, more layers, better tokenizer, gradient accumulation, mixed precision, optimized kernels.
- **Extend it** — new attention variants, different positional encodings, additional heads.
- **Use it** — as a teaching tool, as a baseline, or as a starting point for experiments.
- **Port it** — to other languages that share PHP's constraints.

Whether you are a researcher, a student, a small team, or a large organization, the code is here for you to read, modify, and build on.

## 10. Links

- Website: **jibay.ir**
- Team: **JibayAI**
- Model: **Jibay Nano**

---

# فارسی

## ۱. معرفی

**Jibay Nano** یک مدل زبانی واقعی و کاملاً کارآمد از خانوادهٔ ترنسفورمر (decoder-only) است که **به‌طور کامل با PHP** نوشته شده است — بدون هیچ کتابخانهٔ خارجی یادگیری ماشین، بدون Python، بدون TensorFlow، بدون PyTorch، و بدون GPU. همه‌چیز — آموزش توکنیزر، پاس رو به جلو، پاس رو به عقب، محاسبهٔ گرادیان، بهینه‌ساز Adam، ذخیره‌سازی وزن‌ها، و تولید متن به‌صورت autoregressive — از صفر در PHP پیاده‌سازی شده است.

این پروژه یک پوشش (wrapper) روی یک API آماده نیست، و یک جدول جستجوی پاسخ‌های از پیش نوشته‌شده هم نیست. یک شبکهٔ عصبی واقعی و کوچک است که با backpropagation از داده یاد می‌گیرد و خروجی را با نمونه‌برداری از یک توزیع احتمال روی واژگان خود تولید می‌کند.

هدف این پروژه اثبات یک نکته است: **می‌توان یک معماری واقعی هوش مصنوعی را در PHP ساخت**. PHP زبان طراحی‌شده برای محاسبات عددی نیست و بستر مناسبی برای یادگیری ماشین در مقیاس بزرگ محسوب نمی‌شود. اما «مناسب نبودن» با «غیرممکن بودن» یکی نیست. Jibay Nano نشان می‌دهد که معماری، ریاضیات، و حلقهٔ آموزش می‌توانند در یک فایل PHP زندگی کنند و روی یک هاست معمولی وب اجرا شوند.

## ۲. مشخصات مدل

| ویژگی | مقدار |
|---|---|
| نام مدل | Jibay Nano |
| کلاس معماری | `JibayNanoForCausalLM` |
| نوع مدل | مدل زبانی علّی decoder-only (سبک GPT) |
| پارامترها | **۸٬۳۹۴** وزن آموزش‌پذیر |
| تعداد لایه‌ها (`n_layers`) | **۲** بلوک ترنسفورمر |
| بُعد پنهان (`d_model`) | **۸** |
| تعداد هدهای توجه (`n_heads`) | **۲** (بُعد هر هد = ۴) |
| بُعد شبکهٔ پیش‌خور (`ff_dim`) | **۱۶** |
| تابع فعال‌سازی | GELU |
| نرمال‌سازی | Pre-LayerNorm |
| نوع توجه | Causal multi-head self-attention |
| اندازهٔ واژگان | **۲۵۷** توکن |
| تعداد ادغام‌های BPE | **۱۴۷** |
| طول کانتکست (`max_position_embeddings`) | **۶۸** توکن |
| حداکثر توکن خروجی (`generation_limit`) | **۷۵** توکن |
| بهینه‌ساز | Adam (β₁ = 0.9، β₂ = 0.999، ε = 1e-8) |
| نرخ یادگیری پایه | 0.001 (با warm-up در ۵ گام اول) |
| اندازهٔ batch | ۸ |
| تعداد نمونه‌های آموزش | **۲۱۰** جفت پرسش و پاسخ فارسی |
| تعداد epochها | ۱۰ |
| تعداد گام‌های آموزش | ۲۷۰ |
| زمان آموزش | ~۷۷۶ ثانیه (CPU، تک‌پردازش) |
| loss نهایی آموزش | 5.709 |
| سر اضافی | Multi-Token Prediction (MTP) با rejection sampling |
| فرمت وزن‌ها | باینری float32 با هدر magic + SHA-256 معماری |

## ۳. معماری

Jibay Nano از طراحی استاندارد **ترنسفورمر decoder-only** پیروی می‌کند — همان خانوادهای که GPT به آن تعلق دارد — اما به کوچک‌ترین اندازهٔ ممکن که هنوز به‌درستی کار می‌کند، کاهش یافته است.

**خط لولهٔ پاس رو به جلو:**

1. **Embedding توکن** — هر شناسهٔ توکن ورودی به یک بردار ۸ بُعدی نگاشت می‌شود.
2. **Embedding موقعیت** — یک بردار موقعیت آموزش‌پذیر به هر توکن اضافه می‌شود.
3. **بلوک ترنسفورمر × ۲** که هرکدام شامل موارد زیر است:
   - Pre-LayerNorm → causal multi-head self-attention (۲ هد) → اتصال residual
   - Pre-LayerNorm → شبکهٔ پیش‌خور (۸ → ۱۶ → ۸) با GELU → اتصال residual
4. **LayerNorm نهایی**
5. **سر LM** — تصویر خطی از ۸ به ۲۵۷ لوجیت.
6. **سر MTP** — یک تصویر خطی دوم برای هدف کمکی Multi-Token Prediction.

**انتخاب‌های کلیدی طراحی:**

- **ماسک علّی (causal masking)** — موقعیت *t* فقط می‌تواند به موقعیت‌های ≤ *t* توجه کند؛ این تولید autoregressive را الزامی می‌کند.
- **Pre-LayerNorm** — نرمال‌سازی قبل از هر زیرلایه انجام می‌شود که آموزش مدل‌های کوچک را پایدار می‌کند.
- **فعال‌سازی GELU** — نرم‌تر از ReLU و استاندارد در ترنسفورمرهای مدرن.
- **بدون weight tying** — ماتریس embedding ورودی و تصویر خروجی جدا هستند؛ این پیاده‌سازی و اشکال‌زدایی را ساده‌تر می‌کند.

## ۴. توکنیزر

توکنیزر یک مدل **Byte-Pair Encoding (BPE)** است که از صفر روی پیکرهٔ آموزش، آموزش دیده:

- توکن‌های ویژه: `<|pad|>` (0)، `<|bos|>` (1)، `<|eos|>` (2)، `<|unk|>` (3)
- pre-tokenizer آگاه به یونیکد که حروف، ارقام و آندرلاین را در تکه‌های واژگانی گروه‌بندی می‌کند
- فاصله، تب و خط جدید به نشانگرهای `Ġ`، `ĉ` و `↵` نگاشت می‌شوند تا فاصله‌ها بتوانند بخشی از ادغام‌های BPE باشند
- **۲۵۷** ورودی واژگان نهایی که از **۱۴۷** عملیات ادغام حاصل شده‌اند
- پشتیبانی کامل از حروف فارسی، نیم‌فاصله (`‌`)، ارقام فارسی و حروف لاتین

توکنیزر در اولین ساخت مدل به‌طور خودکار آموزش می‌بیند و در فایل‌های `vocab.json`، `merges.txt`، `tokenizer_config.json` و `tokenizer.json` ذخیره می‌شود.

## ۵. نحوهٔ کار

### ۵.۱ آموزش

1. `dataset.json` بارگذاری می‌شود (۲۱۰ نمونهٔ پرسش و پاسخ فارسی).
2. توکنیزر BPE روی پیکره آموزش می‌بیند.
3. معماری مدل به‌طور خودکار از آمار دیتاست استخراج می‌شود (طول کانتکست از صدک ۹۰ طول دنباله‌ها، محدودیت تولید از صدک ۹۰ طول پاسخ‌ها، اندازهٔ مدل از تعداد نمونه‌ها).
4. برای هر batch، پاس رو به جلو لوجیت‌ها را محاسبه می‌کند؛ پاس رو به عقب گرادیان‌ها را **به‌صورت تحلیلی** برای همهٔ وزن‌ها محاسبه می‌کند (attention، LayerNorm، GELU، لایه‌های خطی، embeddingها).
5. گرادیان‌ها clip می‌شوند و سپس با Adam اعمال می‌شوند.
6. هم loss اصلی پیش‌بینی توکن بعدی و هم loss کمکی **MTP (Multi-Token Prediction)** به‌طور مشترک بهینه می‌شوند — مدل به‌طور اضافی یاد می‌گیرد توکن **دو** قدم جلوتر را پیش‌بینی کند، که در زمان استنتاج برای speculative decoding به کار می‌رود.
7. Checkpoint بعد از هر batch ذخیره می‌شود تا آموزش پس از crash یا timeout از همان‌جا ادامه یابد.
8. Early stopping با patience ۳ epoch بر loss نظارت می‌کند.

### ۵.۲ استنتاج

1. پرامپت توکنایز می‌شود: `User: <question>\nAssistant:` با پیشوند `<|bos|>`.
2. پنجرهٔ کانتکست به ۶۸ توکن آخر محدود می‌شود.
3. برای هر توکن تولیدی:
   - پاس رو به جلو لوجیت‌های اصلی و MTP را تولید می‌کند.
   - نمونه‌برداری با **temperature**، **top-k** و **top-p** اعمال می‌شود.
   - سر MTP یک توکن پیش‌نویس برای موقعیت *t+2* پیشنهاد می‌دهد.
   - یک مرحلهٔ **rejection sampling** توزیع پیش‌نویس را با توزیع هدف مقایسه می‌کند و پیش‌نویس را می‌پذیرد یا رد می‌کند. این یک تکنیک speculative decoding است که در مدل‌های بزرگ بدون تغییر توزیع خروجی، تولید را سریع‌تر می‌کند.
4. توکن‌های ویژه (`<|bos|>`، `<|pad|>`، `<|unk|>` و `<|eos|>` در دو گام اول) ماسک می‌شوند.
5. تولید با `<|eos|>` یا رسیدن به سقف ۷۵ توکن متوقف می‌شود.

## ۶. فایل‌های پروژه

| فایل | کاربرد |
|---|---|
| `index.php` | تمام مدل: توکنیزر، آموزش، استنتاج، رابط HTTP |
| `dataset.json` | داده‌های آموزش (۲۱۰ جفت پرسش و پاسخ فارسی، متن‌باز) |
| `config.json` | پیکربندی مدل + فرادادهٔ آموزش |
| `model_config.json` | آینهٔ `config.json` |
| `vocab.json` | واژگان BPE (توکن → id) |
| `merges.txt` | قواعد ادغام BPE |
| `tokenizer_config.json` | فرادادهٔ توکنیزر |
| `tokenizer.json` | توکنیزر کامل در چیدمان سازگار با Hugging Face |
| `model_weights.bin` | وزن‌های آموزش‌دیده (باینری float32 + هدر) |
| `training_state.json` | پیشرفت آموزش به‌صورت خوانا برای انسان |
| `training_checkpoint.tmp` | checkpoint آموزش در حال انجام (موقت) |
| `training.lock` | قفل فایل برای جلوگیری از آموزش هم‌زمان |

## ۷. استفاده

`index.php` را در هر محیط PHP اجرا کنید (PHP 8.1+ توصیه می‌شود، بدون نیاز به افزونه).

- **گزارش مدل:** `index.php` را بدون پارامتر باز کنید.
- **پرسیدن سؤال:** `index.php?t=سلام`
- **غیرفعال‌کردن MTP:** `index.php?t=سلام&mtp=0`
- **تنظیم نمونه‌برداری:** `index.php?t=سلام&temperature=0.7&top_k=20&top_p=0.85`
- **ادامهٔ آموزش:** `index.php?train=1`

اگر `config.json` و `model_weights.bin` موجود و سازگار باشند، مدل بدون آموزش مجدد بارگذاری می‌شود. اگر دیتاست تغییر کند، وزن‌های موجود به‌طور خودکار نامعتبر می‌شوند و مدل از نو ساخته می‌شود.

## ۸. این مدل چه هست و چه نیست

**هست:**
- یک شبکهٔ عصبی ترنسفورمر واقعی با وزن‌های واقعیِ آموزش‌دیده.
- یک پیاده‌سازی واقعی از backpropagation با گرادیان واقعی.
- یک اثبات کاری بر این که معماری یک مدل زبانی مدرن را می‌توان در PHP ساخت و آموزش داد.
- یک معماری متن‌باز برای پژوهش، آموزش و آزمایش.

**نیست:**
- یک دستیار دانا. انتظار نداشته باشید به پرسش‌های علمی، تاریخی، جغرافیایی یا خبری روز پاسخ درست بدهد.
- جایگزینی برای GPT، LLaMA، Claude یا هر مدل زبانی بزرگ دیگر.
- مدلی آموزش‌دیده روی پیکره‌ای بزرگ و متنوع. این مدل ۲۱۰ نمونه و ۸٬۳۹۴ پارامتر دارد.

loss نهایی آموزش **۵٫۷۱** است، و loss نظری یک مدل کاملاً تصادفی روی واژگان ۲۵۷ توکنی **ln(257) ≈ 5.55** است. این یعنی مدل *مقداری* ساختار یاد گرفته — به‌طور قابل‌اندازه‌گیری بهتر از حدس زدن است — اما ظرفیت آن بسیار کمتر از آن است که به‌عنوان دستیار عمومی مفید باشد. **این طبیعی و مورد انتظار است.**

## ۹. چرا PHP، و چرا منتشرش کردیم

PHP زبانی نیست که برای محاسبات عددی طراحی شده باشد. برای عملیات ماتریسی بهینه نشده، نوع دادهٔ tensor توکار ندارد، و زمان اجرایش بسیار کندتر از C، Rust یا CUDA است. هر کتابخانهٔ جدی یادگیری ماشین در جهان به زبان دیگری نوشته شده است.

**با این حال Jibay Nano را ساختیم — تا نشان دهیم که می‌توانیم.**

این معماری را آشکارا منتشر می‌کنیم چون باور داریم ارزش آن در وزن‌ها نیست (که بسیار کوچک‌اند)، بلکه در **طراحی** است: در این واقعیت که یک مدل کامل سبک GPT — توکنیزر، ترنسفورمر، backpropagation، Adam، MTP، rejection sampling، سریال‌سازی وزن، checkpointing — می‌تواند در یک فایل PHP نوشته شود و با خواندنش فهمیده شود.

از جامعه دعوت می‌کنیم تا:

- **بهبودش دهند** — `d_model` بزرگ‌تر، لایه‌های بیشتر، توکنیزر بهتر، انباشت گرادیان، دقت مختلط، کرنل‌های بهینه‌تر.
- **گسترشش دهند** — انواع جدید attention، کدگذاری‌های موقعیتی متفاوت، هدهای اضافی.
- **از آن استفاده کنند** — به‌عنوان ابزار آموزشی، به‌عنوان خط پایه، یا به‌عنوان نقطهٔ شروع برای آزمایش.
- **پورتش کنند** — به زبان‌های دیگری که محدودیت‌های PHP را شریک‌اند.

چه پژوهشگر باشید، چه دانشجو، چه یک تیم کوچک، چه یک سازمان بزرگ، کد اینجاست تا بخوانید، تغییر دهید و رویش بسازید.

## ۱۰. پیوندها

- وب‌سایت: **jibay.ir**
- تیم: **JibayAI**
- مدل: **Jibay Nano**

---

**Jibay Nano** — a real Transformer in PHP. Built to prove it can be built.
**Jibay Nano** — یک ترنسفورمر واقعی در PHP. ساخته شد تا ثابت شود می‌توان ساخت.