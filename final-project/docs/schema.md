# Data Schema

## القاعدة الذهبية
كل ملف بياخد **كل أعمدة الملف اللي قبله ويضيف أعمدة جديدة بس**.
ممنوع حذف عمود أو تغيير اسمه.

## ترتيب الملفات

```
raw_data.csv → cleaned_a.csv → cleaned_final.csv → model_output.csv
   (P1)            (P2)             (P3)               (P4)
```

## أعمدة ملفات التقييمات

| العمود | مين بيضيفه | القيمة |
|---|---|---|
| `review_id` | Person 1 | رقم التقييم من Google Play |
| `app_name` | Person 1 | اسم التطبيق بالإنجليزي زي ما في الـ dictionary (مثال: `Vodafone Egypt`) |
| `domain` | Person 1 | `Telecom` / `Payments` / `Delivery` |
| `rating` | Person 1 | رقم من 1 لـ 5 |
| `date` | Person 1 | بالشكل `2026-03-14` |
| `review_text` | Person 1 | النص الأصلي زي ما هو |
| `review_type` | Person 1 | `Negative` (1–2) / `Neutral` (3) / `Positive` (4–5) |
| `normalized_text` | Person 2 | النص بعد التطبيع اللغوي |
| `is_franco` | Person 2 | `True` / `False` |
| `final_text` | Person 3 | النص النضيف اللي الموديل هيشتغل عليه |
| `topic_label` | Person 4 | اسم الموضوع (عام لكل شركات القطاع) |
| `severity` | Person 4 | `High` / `Medium` / `Low` للسلبي بس، وفاضي لو مش سلبي |

## ملفات الشخص 5

| الملف | الأعمدة |
|---|---|
| `competitive_output.csv` | `domain, app_name, topic_label, total_reviews, complaint_pct, praise_pct` |
| `gaps_summary.csv` | `domain, topic_label, weak_app, weak_pct, strong_app, strong_pct, gap_size, sentence` |

- `complaint_pct` = نسبة `Negative` من كل تعليقات (الشركة + الموضوع).
- `praise_pct` = نسبة `Positive` من كل تعليقات (الشركة + الموضوع).

## قواعد بسيطة

- أسماء الأعمدة إنجليزي، حروف صغيرة، وبينها `_`.
- القيم المسموحة تتكتب بالضبط (`Negative` مش `negative`).
- احفظ أي CSV بالشكل ده: `df.to_csv(path, index=False, encoding="utf-8-sig")`
- ملف الداتا بيعدله صاحبه بس، والباقي بيقروه.
- لو محتاج تغيير في الأعمدة، كلّم الفريق الأول وعدّل الملف ده.
