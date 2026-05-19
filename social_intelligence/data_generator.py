"""
Data generation module for the Social Intelligence Platform.
Generates realistic simulated social media data with Arabic content
for demonstration and testing purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import hashlib

ARABIC_TOPICS = [
    "الاقتصاد الرقمي", "التحول الوطني", "رؤية 2030", "الطاقة المتجددة",
    "التعليم الإلكتروني", "الذكاء الاصطناعي", "الأمن السيبراني", "السياحة",
    "الصحة العامة", "البنية التحتية", "الاستثمار الأجنبي", "ريادة الأعمال",
    "التقنية المالية", "المدن الذكية", "البيئة والاستدامة", "الثقافة والفنون",
    "الرياضة", "النقل العام", "الإسكان", "سوق العمل"
]

ARABIC_HASHTAGS = [
    "#التحول_الرقمي", "#رؤية_2030", "#الاقتصاد_السعودي", "#التعليم",
    "#الصحة", "#السياحة_السعودية", "#الطاقة_المتجددة", "#الذكاء_الاصطناعي",
    "#ريادة_الأعمال", "#التقنية", "#الاستثمار", "#المدن_الذكية",
    "#البيئة", "#الثقافة", "#الرياضة_السعودية", "#نيوم",
    "#الترفيه", "#التوظيف", "#الابتكار", "#العمل_عن_بعد",
    "#أمن_المعلومات", "#الحوكمة", "#الشفافية", "#التنمية_المستدامة"
]

PLATFORMS = ["Twitter/X", "Facebook", "Instagram", "YouTube", "TikTok", "Reddit", "News Sites"]

REGIONS = [
    "الرياض", "جدة", "مكة المكرمة", "المدينة المنورة", "الدمام",
    "الخبر", "تبوك", "أبها", "نجران", "جازان",
    "القاهرة", "دبي", "أبوظبي", "الدوحة", "الكويت",
    "عمّان", "بيروت", "بغداد", "الرباط", "تونس"
]

LANGUAGES = ["العربية", "English", "العربية (لهجة خليجية)", "العربية (لهجة مصرية)", "Fran\u00e7ais"]

SENTIMENT_LABELS = ["إيجابي", "سلبي", "محايد"]
STANCE_LABELS = ["مؤيد", "معارض", "ساخر", "محايد"]

SAMPLE_TEXTS = {
    "إيجابي": [
        "مبادرة رائعة تدعم التحول الرقمي في المملكة وتفتح آفاقاً جديدة للشباب",
        "نتائج مبهرة للاقتصاد الرقمي هذا العام مع نمو ملحوظ في جميع القطاعات",
        "خطوة متقدمة نحو تحقيق أهداف الرؤية والتنمية المستدامة",
        "برنامج تدريبي متميز يؤهل الكوادر الوطنية لسوق العمل المستقبلي",
        "استثمارات ضخمة في البنية التحتية الرقمية تعزز مكانة المملكة عالمياً",
        "إنجاز تقني جديد يضع المملكة في مصاف الدول المتقدمة تقنياً",
        "تطور ملحوظ في قطاع التعليم الإلكتروني يستحق الإشادة والتقدير",
        "نمو اقتصادي قوي يعكس نجاح السياسات الاقتصادية الحديثة",
    ],
    "سلبي": [
        "ارتفاع تكاليف المعيشة يشكل ضغطاً كبيراً على المواطنين",
        "تأخر في تنفيذ بعض المشاريع الحيوية يثير قلق المتابعين",
        "فجوة واضحة بين الخطاب الرسمي والواقع الفعلي في بعض القطاعات",
        "معدلات البطالة لا تزال مرتفعة رغم الجهود المبذولة في التوظيف",
        "ضعف البنية التحتية في بعض المناطق يعيق التنمية المتوازنة",
        "غياب الشفافية في بعض القرارات يثير تساؤلات مشروعة",
        "ارتفاع أسعار العقارات يحد من قدرة الشباب على التملك",
        "تراجع جودة الخدمات في بعض القطاعات الحكومية",
    ],
    "محايد": [
        "تقرير جديد يرصد مؤشرات الاقتصاد الرقمي في المنطقة",
        "مؤتمر دولي يناقش مستقبل التقنية والذكاء الاصطناعي",
        "دراسة أكاديمية تحلل تأثير وسائل التواصل على الرأي العام",
        "إحصائيات جديدة حول استخدام الإنترنت في المنطقة العربية",
        "ندوة تناقش التحديات والفرص في سوق العمل الرقمي",
        "بيان رسمي يوضح تفاصيل القرارات الاقتصادية الأخيرة",
        "تقرير دوري عن أداء القطاعات الاقتصادية خلال الربع الأخير",
        "استطلاع رأي جديد حول أولويات المواطنين في المرحلة القادمة",
    ]
}

ACCOUNT_TYPES = ["حساب شخصي", "حساب رسمي", "حساب إعلامي", "حساب تجاري", "حساب مجهول"]

NARRATIVE_THEMES = [
    "الإصلاح الاقتصادي", "التحول الرقمي", "جودة الحياة", "الاستدامة البيئية",
    "تمكين الشباب", "تمكين المرأة", "الأمن الغذائي", "التنويع الاقتصادي",
    "الحوكمة والشفافية", "الابتكار التقني", "الهوية الثقافية", "التعاون الدولي"
]


def _make_id(seed_str: str) -> str:
    return hashlib.md5(seed_str.encode()).hexdigest()[:12]


def generate_accounts(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    accounts = []
    for i in range(n):
        acct_type = rng.choice(ACCOUNT_TYPES, p=[0.40, 0.15, 0.15, 0.15, 0.15])
        is_bot = rng.random() < 0.08
        followers = int(rng.lognormal(mean=8, sigma=2))
        following = int(rng.lognormal(mean=7, sigma=1.5))
        influence_score = min(100, max(0, int(
            np.log1p(followers) * 5 + rng.normal(0, 5)
        )))
        accounts.append({
            "account_id": _make_id(f"acct_{i}_{seed}"),
            "username": f"user_{i:04d}",
            "display_name": f"مستخدم {i}" if rng.random() < 0.7 else f"User {i}",
            "account_type": acct_type,
            "platform": rng.choice(PLATFORMS),
            "region": rng.choice(REGIONS),
            "followers": followers,
            "following": following,
            "influence_score": influence_score,
            "is_verified": rng.random() < 0.12,
            "is_bot_suspect": is_bot,
            "created_date": datetime(2018, 1, 1) + timedelta(days=int(rng.uniform(0, 2000))),
            "activity_level": rng.choice(["عالي", "متوسط", "منخفض"], p=[0.2, 0.5, 0.3]),
        })
    return pd.DataFrame(accounts)


def generate_posts(n: int = 5000, days: int = 90, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    accounts = generate_accounts(200, seed)
    posts = []
    base_date = datetime.now() - timedelta(days=days)

    trend_peak_day = rng.randint(min(20, days // 3), max(days - 20, days // 3 + 1))
    crisis_day = rng.randint(min(10, days // 4), max(days - 10, days // 4 + 1))

    for i in range(n):
        day_offset = rng.randint(0, days)
        hour = rng.choice(range(24), p=_hour_distribution())
        minute = rng.randint(0, 60)

        timestamp = base_date + timedelta(days=int(day_offset), hours=int(hour), minutes=int(minute))
        sentiment = rng.choice(SENTIMENT_LABELS, p=_sentiment_distribution(day_offset, crisis_day))
        text = rng.choice(SAMPLE_TEXTS[sentiment])
        topic = rng.choice(ARABIC_TOPICS)
        hashtags_count = rng.poisson(1.5)
        selected_hashtags = list(rng.choice(ARABIC_HASHTAGS, size=min(hashtags_count, 5), replace=False))
        account = accounts.iloc[rng.randint(0, len(accounts))]
        engagement_base = np.log1p(account["followers"]) * rng.uniform(0.5, 2.0)

        is_near_peak = abs(day_offset - trend_peak_day) < 5
        volume_multiplier = 2.5 if is_near_peak else 1.0

        likes = int(engagement_base * rng.uniform(1, 20) * volume_multiplier)
        retweets = int(engagement_base * rng.uniform(0.1, 5) * volume_multiplier)
        replies = int(engagement_base * rng.uniform(0.05, 2) * volume_multiplier)

        narrative = rng.choice(NARRATIVE_THEMES)

        posts.append({
            "post_id": _make_id(f"post_{i}_{seed}"),
            "account_id": account["account_id"],
            "username": account["username"],
            "display_name": account["display_name"],
            "platform": account["platform"],
            "region": account["region"],
            "timestamp": timestamp,
            "date": timestamp.date(),
            "hour": hour,
            "text": text,
            "topic": topic,
            "hashtags": selected_hashtags,
            "hashtags_str": " ".join(selected_hashtags),
            "sentiment": sentiment,
            "sentiment_score": _sentiment_score(sentiment, rng),
            "stance": rng.choice(STANCE_LABELS, p=_stance_from_sentiment(sentiment)),
            "language": rng.choice(LANGUAGES, p=[0.55, 0.20, 0.10, 0.10, 0.05]),
            "likes": likes,
            "retweets": retweets,
            "replies": replies,
            "engagement": likes + retweets * 2 + replies * 3,
            "reach": int((likes + retweets) * rng.uniform(5, 50)),
            "narrative": narrative,
            "is_bot_content": account["is_bot_suspect"],
            "credibility_score": rng.uniform(0.3, 1.0) if not account["is_bot_suspect"] else rng.uniform(0.05, 0.4),
            "impact_score": rng.uniform(0, 100),
            "is_coordinated": rng.random() < 0.05,
            "virality_score": rng.uniform(0, 100),
        })

    df = pd.DataFrame(posts)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generate_alerts(posts_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    alerts = []
    alert_types = [
        ("ارتفاع مفاجئ في الحجم", "حرج", "volume_spike"),
        ("تحول في المشاعر", "عالي", "sentiment_shift"),
        ("هاشتاغ جديد صاعد", "متوسط", "new_hashtag"),
        ("نشاط مشبوه مكتشف", "حرج", "suspicious_activity"),
        ("حملة منظمة محتملة", "عالي", "coordinated_campaign"),
        ("سردية ناشئة", "متوسط", "emerging_narrative"),
        ("تحول في الخطاب", "عالي", "discourse_shift"),
        ("ذروة تفاعل غير عادية", "متوسط", "engagement_peak"),
    ]

    dates = posts_df["date"].unique()
    for date in rng.choice(dates, size=min(30, len(dates)), replace=False):
        n_alerts = rng.randint(1, 4)
        for _ in range(n_alerts):
            alert_type = alert_types[rng.randint(0, len(alert_types))]
            alerts.append({
                "alert_id": _make_id(f"alert_{date}_{rng.randint(0,10000)}"),
                "timestamp": pd.Timestamp(date) + timedelta(hours=rng.randint(0, 24)),
                "type": alert_type[0],
                "severity": alert_type[1],
                "category": alert_type[2],
                "description": f"تم رصد {alert_type[0]} في تاريخ {date}",
                "topic": rng.choice(ARABIC_TOPICS),
                "status": rng.choice(["جديد", "قيد المراجعة", "تمت المعالجة"], p=[0.5, 0.3, 0.2]),
                "assigned_to": rng.choice(["محلل 1", "محلل 2", "محلل 3", "غير مخصص"]),
            })

    return pd.DataFrame(alerts).sort_values("timestamp", ascending=False).reset_index(drop=True)


def generate_narratives(posts_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    narratives = []
    for theme in NARRATIVE_THEMES:
        theme_posts = posts_df[posts_df["narrative"] == theme]
        if len(theme_posts) == 0:
            continue
        start_date = theme_posts["date"].min()
        end_date = theme_posts["date"].max()
        avg_sentiment = theme_posts["sentiment_score"].mean()
        total_engagement = theme_posts["engagement"].sum()
        post_count = len(theme_posts)

        evolution_phase = rng.choice(["ناشئة", "متنامية", "في الذروة", "متراجعة", "مستقرة"])
        narratives.append({
            "narrative_id": _make_id(f"narr_{theme}"),
            "theme": theme,
            "start_date": start_date,
            "end_date": end_date,
            "post_count": post_count,
            "avg_sentiment": avg_sentiment,
            "total_engagement": total_engagement,
            "evolution_phase": evolution_phase,
            "impact_score": rng.uniform(30, 100),
            "spread_velocity": rng.uniform(0, 100),
            "top_hashtags": list(rng.choice(ARABIC_HASHTAGS, size=3, replace=False)),
            "related_topics": list(rng.choice(ARABIC_TOPICS, size=3, replace=False)),
            "risk_level": rng.choice(["منخفض", "متوسط", "عالي", "حرج"], p=[0.4, 0.3, 0.2, 0.1]),
        })

    return pd.DataFrame(narratives)


def generate_network_data(accounts_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    edges = []
    n_accounts = len(accounts_df)
    n_edges = n_accounts * 3

    for _ in range(n_edges):
        source_idx = rng.randint(0, n_accounts)
        target_idx = rng.randint(0, n_accounts)
        if source_idx == target_idx:
            continue
        interaction_type = rng.choice(["retweet", "reply", "mention", "quote"], p=[0.4, 0.25, 0.25, 0.1])
        edges.append({
            "source": accounts_df.iloc[source_idx]["account_id"],
            "source_name": accounts_df.iloc[source_idx]["username"],
            "target": accounts_df.iloc[target_idx]["account_id"],
            "target_name": accounts_df.iloc[target_idx]["username"],
            "interaction_type": interaction_type,
            "weight": rng.randint(1, 20),
        })

    return pd.DataFrame(edges)


def _hour_distribution():
    hours = np.array([
        0.01, 0.005, 0.003, 0.002, 0.002, 0.005,
        0.01, 0.02, 0.04, 0.06, 0.07, 0.07,
        0.06, 0.05, 0.05, 0.05, 0.06, 0.07,
        0.07, 0.08, 0.08, 0.06, 0.04, 0.02
    ])
    return hours / hours.sum()


def _sentiment_distribution(day_offset, crisis_day):
    if abs(day_offset - crisis_day) < 3:
        return [0.15, 0.55, 0.30]
    return [0.35, 0.25, 0.40]


def _sentiment_score(sentiment, rng):
    if sentiment == "إيجابي":
        return rng.uniform(0.5, 1.0)
    elif sentiment == "سلبي":
        return rng.uniform(-1.0, -0.3)
    else:
        return rng.uniform(-0.3, 0.3)


def _stance_from_sentiment(sentiment):
    if sentiment == "إيجابي":
        return [0.6, 0.05, 0.05, 0.3]
    elif sentiment == "سلبي":
        return [0.05, 0.5, 0.2, 0.25]
    else:
        return [0.15, 0.15, 0.1, 0.6]


@pd.api.extensions.register_dataframe_accessor("si")
class SocialIntelligenceAccessor:
    """Pandas accessor for common social intelligence operations."""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def daily_volume(self):
        return self._obj.groupby("date").size().reset_index(name="count")

    def sentiment_over_time(self):
        return self._obj.groupby(["date", "sentiment"]).size().unstack(fill_value=0)

    def top_hashtags(self, n=20):
        all_tags = []
        for tags in self._obj["hashtags"]:
            if isinstance(tags, list):
                all_tags.extend(tags)
        return pd.Series(all_tags).value_counts().head(n)

    def top_accounts(self, n=20, by="engagement"):
        return self._obj.groupby("username")[by].sum().sort_values(ascending=False).head(n)

    def platform_distribution(self):
        return self._obj["platform"].value_counts()

    def region_distribution(self):
        return self._obj["region"].value_counts()
