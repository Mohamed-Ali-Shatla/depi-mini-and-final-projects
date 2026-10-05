# Competitive Analysis

تحليل تنافسي لتقييمات تطبيقات Google Play المصرية (اتصالات، مدفوعات، توصيل).
بنسحب التقييمات، ننضّفها، نصنّفها بالموديل حسب الموضوع والخطورة، وبعدين نكتشف الفجوات بين الشركات المنافسة ونعرضها في Dashboard.

---

## الفريق

| # | الدور | الاسم | حساب GitHub | Branch |
|---|---|---|---|---|
| 1 | جمع الداتا | | | `person1-collection` |
| 2 | التطبيع اللغوي | | | `person2-normalization` |
| 3 | التنضيف و EDA | | | `person3-cleaning` |
| 4 | الموديل | | | `person4-model` |
| 5 | التحليل التنافسي | | | `person5-competitive` |
| 6 | Dashboard | | | `person6-dashboard` |

---

## الملفات المهمة

| الملف | فيه إيه |
|---|---|
| [`docs/schema.md`](docs/schema.md) | أعمدة كل ملفات الداتا (اقراه قبل ما تكتب أي كود) |
| [`docs/`](docs/) | ملف المهام (الملف 1) والتايم لاين (الملف 2) |

---

## شكل الريبو

```
competitive-analysis/
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   └── schema.md
│
├── data/
│   ├── raw/          raw_data.csv              # person 1
│   ├── cleaned/      cleaned_a.csv             # person 2
│   │                 cleaned_final.csv         # person 3
│   └── processed/    model_output.csv          # person 4
│                     competitive_output.csv    # person 5
│                     gaps_summary.csv          # person 5
│
├── notebooks/
│   ├── 1_collection/       # person 1
│   ├── 2_normalization/    # person 2
│   ├── 3_cleaning_eda/     # person 3
│   ├── 4_model/            # person 4
│   └── 5_competitive/      # person 5
│
└── dashboard/
    └── app.py              # person 6
```

**كل واحد بيشتغل في فولدره بس.** ده بيمنع تقريبًا كل الـ conflicts.

---

## أول مرة تشتغل

### 1. حمّل الريبو

```bash
git clone <رابط-الريبو>
cd competitive-analysis
```

### 2. اعمل virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. نزّل المكتبات

```bash
pip install -r requirements.txt
```

> المكتبات الخاصة بالشخص 4 تقيلة. لو مش إنت، شوف التعليقات جوه `requirements.txt` ونزّل اللي محتاجه بس.

---

## طريقة الشغل اليومية على Git

`main` فيه الشغل الجاهز بس. **محدش بيشتغل عليه مباشرة.**

### أول مرة (مرة واحدة)

```bash
git checkout -b person1-collection      # غيّر الاسم لاسم branch بتاعك
```

### كل يوم

```bash
git pull origin main                    # قبل ما تبدأ
# ... اشتغل ...
git add .
git commit -m "وصف قصير للي عملته"
git push -u origin person1-collection
```

### لما تخلص حاجة

1. افتح الريبو على GitHub واضغط **Compare & pull request**.
2. واحد من الفريق يراجعها.
3. يضغط **Merge**.

---

## قواعد الفريق

- اقرا `docs/schema.md`. ممنوع تحذف عمود أو تغيّر اسمه، وتضيف أعمدتك بس.
- احفظ أي CSV كده: `df.to_csv(path, index=False, encoding="utf-8-sig")`
- ملف الداتا بيعدّله صاحبه بس، والباقي بيقروه.
- لو خلصت بدري، سلّم فورًا. أي يوم بيتوفر بيروح للي بعدك.
- لو اتأخرت 3 أيام أو أكتر، بلّغ الفريق في نفس اليوم.
- لو محتاج تغيّر في الـ Schema، كلّم الفريق الأول وعدّل `docs/schema.md`.
- متعملش commit للـ `venv/` (موجود في `.gitignore`).

---

## القطاعات

| القطاع | `domain` |
|---|---|
| اتصالات | `Telecom` |
| مدفوعات | `Payments` |
| توصيل | `Delivery` |
