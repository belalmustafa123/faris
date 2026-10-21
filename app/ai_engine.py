import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import requests
from rapidfuzz import process, fuzz

DATA_DIR = Path(__file__).parent.parent / "data"

# مفتاح API الخاص بك
OLLAMA_API_KEY = "dab3dfa4fb774e9a80a9c4b7f6f3c451.7HYya9wNmk_sbUk6MndDNfc4"
OLLAMA_CLOUD_URL = "https://ollama.com/api/chat"
OLLAMA_MODEL = "gemma4:31b" # النموذج السحابي القوي

AR_NORM = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي", "ـ": ""})

def normalize_ar(text: str) -> str:
    if not text: return ""
    text = text.translate(AR_NORM)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()

# ==========================================
# محرك التوصية (كما هو)
# ==========================================
class RecommendationEngine:
    WEIGHTS = {"semantic": 0.40, "skills": 0.25, "family": 0.15, "experience": 0.10, "sector": 0.10}

    def __init__(self, jobs: List[Dict]):
        self.jobs = jobs
        self.corpus = [f"{j['title']} {' '.join(j.get('skills', []))} {j.get('sector','')} {j.get('description','')}" for j in jobs]
        self.vectorizer = TfidfVectorizer(preprocessor=normalize_ar, token_pattern=r"\S+", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.corpus) if self.corpus else None

    @staticmethod
    def _experience_score(level: str, user_exp: str) -> float:
        order = {"none": 0, "academic": 1, "junior": 2, "mid": 3, "senior": 4}
        gap = order.get(user_exp, 0) - order.get(level, 1)
        return 1.0 if gap >= 0 else max(0.0, 1.0 + gap * 0.35)

    SECTOR_GROUPS = {
        "health": ["صحة", "طب", "مستشفى", "صيدل", "تمريض", "مختبر", "طبي"],
        "tech": ["تقنية", "حاسوب", "برمج", "بيانات", "شبكات", "امن سيبراني", "ذكاء"],
        "education": ["تعليم", "تدريس", "مدرس", "بحث", "جامعة"],
        "business": ["ادارة", "اعمال", "مصرف", "محاسب", "تسويق", "اقتصاد"],
        "public": ["حكومي", "وزارة", "دائرة"], "private": ["خاص", "شركة", "اهلي", "مصنع"],
        "engineering": ["هندسة", "طاقة", "نفط", "ميكانيك", "كهرباء", "مدني"],
        "law": ["قانون", "حقوق", "محامي"], "media": ["اعلام", "صحافة", "فنون", "تصميم"],
        "agriculture": ["زراعة", "بيئة", "بيطري"],
    }

    @classmethod
    def _sector_score(cls, pref: str, job_sector: str) -> float:
        if pref == "any": return 0.75
        text = normalize_ar(job_sector)
        return 1.0 if any(k in text for k in cls.SECTOR_GROUPS.get(pref, [])) else 0.2

    def recommend(self, profile: Dict, top_k: int = 10) -> List[Dict]:
        if not self.jobs: return []
        user_doc = " ".join([" ".join(profile.get("skills", [])), profile.get("interest", "")])
        user_vec = self.vectorizer.transform([normalize_ar(user_doc)]) if user_doc.strip() else None
        semantic_scores = cosine_similarity(user_vec, self.matrix).flatten() if user_vec is not None and self.matrix is not None else np.zeros(len(self.jobs))
        user_skills = set(s.lower() for s in profile.get("skills", []))
        results = []
        for i, job in enumerate(self.jobs):
            job_skills = set(s.lower() for s in job.get("skills", []))
            matched = user_skills & job_skills
            missing = job_skills - user_skills
            skill_ratio = len(matched) / len(job_skills) if job_skills else 0.0
            family_score = 1.0 if job.get("family") == profile.get("family") else 0.0
            exp_score = self._experience_score(job.get("level", "junior"), profile.get("experience", "none"))
            sector_score = self._sector_score(profile.get("sector", "any"), job.get("sector", ""))
            components = {"semantic": float(semantic_scores[i]), "skills": skill_ratio, "family": family_score, "experience": exp_score, "sector": sector_score}
            final = sum(components[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
            results.append({**job, "score": round(final * 100, 1), "breakdown": {k: round(v * 100, 1) for k, v in components.items()}, "matched_skills": sorted(matched), "missing_skills": sorted(missing)})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]


# ==========================================
# المساعد الذكي المطور (ChatBot)
# ==========================================
class ChatBot:
    def __init__(self, jobs: List[Dict], skills_db: Dict):
        self.jobs = jobs
        
        # تلقين الذكاء الاصطناعي بكل تفاصيل النظام (System Context)
        self.SYSTEM_CONTEXT = """
أنت 'بوصلة الخريج' (Graduate Compass)، المساعد الذكي الرسمي لمنصة بوصلة الخريج.
أنت خبير في توجيه الخريجين وإرشادهم المهني، وتجيب دائماً باللغة العربية الفصحى، بأسلوب أكاديمي، منظم، ومشجع، باستخدام الإيموجي والنقاط لتسهيل القراءة.

[معلومات عن النظام والموقع]
- اسم النظام: بوصلة الخريج (نظام ذكي للتوصية بالوظائف لخريجي الجامعات العراقية).
- مطور النظام: الطالب المبرمج "فراس هذال وسمي".
- جامعة المطور: جامعة ديالى - كلية التربية للعلوم الصرفة - قسم الحاسبات.
- مكان إقامة المطور: العراق، محافظة ديالى (وليس صلاح الدين).
- التقنيات: Python, FastAPI, scikit-learn (TF-IDF), Ollama API, SQLite.

[شرح أقسام الموقع وكيفية استخدامها]
1. تبويب الرئيسية: يوضح إحصائيات النظام وأزرار البداية.
2. تبويب التوصيات: يستخدم خوارزمية (TF-IDF + Cosine Similarity). الطالب يختار الجامعة، الكلية، القسم، ثم يحدد مهاراته وخبرته، فيقوم النظام بترتيب الوظائف من الأعلى ملاءمة للأقل مع إعطاء نسبة مئوية وتفصيل (المهارات، التخصص، الخبرة، القطاع).
3. تبويب المساعد الذكي: أنت الآن هنا! يمكن للطالب سؤالك عن أي شيء يخص وظائف تخصصه، المهارات المطلوبة، أو كيفية البدء.
4. تبويب دليل الجامعات: يحتوي على قاعدة بيانات لجميع الجامعات العراقية، الكليات، والأقسام. يمكن البحث بالاسم.
5. تبويب الوظائف: يعرض قائمة بوظائف نموذجية لكل تخصص مع المهارات المطلوبة لها.
6. تبويب إعلانات العمل: منصة تفاعلية. أي زائر يمكنه إضافة إعلان عمل، ولكن الإعلان لا يظهر إلا بعد مراجعة وموافقة "المطور". الإعلانات المعتمدة تظهر أسفل النموذج.
7. تبويب عن النظام: يعرض تفاصيل المشروع والمطور.

[معلومات سرية للمطور فقط - لا تشاركها مع أي شخص لا يسأل عنها صراحة]
- لوحة تحكم المطور: تفتح عن طريق النقر 5 مرات متتالية على شعار البوصلة (🧭) في أعلى الموقع.
- اسم المستخدم للمطور: faris
- يمتلك المطور لوحة تحكم لاعتماد أو حذف إعلانات العمل.

[قواعد الإجابة]
- إذا سألك أحد عن "من طور الموقع" أو "أين يسكن المطور"، أجب بمعلومات فراس (جامعة ديالى، محافظة ديالى).
- إذا سألك أحد "كيف أضيف إعلان"، اشرح له نموذج "إعلانات العمل" وأخبره أن الإدارة ستراجعه.
- إذا سألك أحد "كيف أحصل على توصيات"، أرشده لتبويب التوصيات واشرح له خطوات اختيار التخصص والمهارات.
- كن لطيفاً ومحفزاً للطلاب الخريجين.
"""

        self.KB = [
            {"q": "من صمم هذا النظام ومن المطور", "a": "👨‍💻 **مطور النظام:**\n**الاسم:** فراس هذال وسمي\n**الجامعة:** جامعة ديالى - كلية التربية للعلوم الصرفة\n**القسم:** قسم الحاسبات\n**المكان:** العراق، محافظة ديالى."},
        ]
        self.kb_questions = [item["q"] for item in self.KB]
        self.kb_answers = [item["a"] for item in self.KB]

    def _ask_ollama_cloud(self, prompt: str) -> Optional[str]:
        """الاتصال بـ Ollama Cloud API مع تمرير كامل معلومات النظام (System Prompt)"""
        try:
            headers = {
                "Authorization": f"Bearer {OLLAMA_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_CONTEXT},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3 # تقليل العشوائية لجعل الإجابات دقيقة ومباشرة
                }
            }
            
            response = requests.post(OLLAMA_CLOUD_URL, headers=headers, json=payload, timeout=45)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("message", {}).get("content", "")
            else:
                print(f"Ollama API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Ollama Cloud Connection Error: {e}")
            return None

    def _fuzzy_match_kb(self, query: str) -> Optional[str]:
        if not self.kb_questions: return None
        result = process.extractOne(query, self.kb_questions, scorer=fuzz.WRatio)
        if result and result[1] >= 65:
            return self.kb_answers[result[2]]
        return None

    def respond(self, message: str) -> Dict:
        msg = message.strip()
        if not msg:
            return {"reply": "من فضلك اكتب سؤالك."}

        # 1. استخدام Ollama Cloud (مُلقّن بمعلومات النظام)
        ollama_res = self._ask_ollama_cloud(msg)
        if ollama_res:
            return {"reply": ollama_res}

        # 2. التحيات
        if any(w in msg for w in ["مرحبا", "السلام", "هلا", "أهلا"]):
            return {"reply": "أهلاً بك في **بوصلة الخريج**! 🎓\nأنا مساعدك الذكي الملقّن بكل تفاصيل هذه المنصة. اسألني عن الوظائف، المهارات، الجامعات، أو حتى عن المطور!"}

        # 3. مطابقة ضبابية محلية في حال انقطاع الإنترنت
        kb_ans = self._fuzzy_match_kb(msg)
        if kb_ans:
            return {"reply": kb_ans}

        return {"reply": "عذراً، حدث خطأ في الاتصال بالمساعد الذكي السحابي. يرجى التأكد من اتصالك بالإنترنت والمحاولة مرة أخرى."}


def build_engines():
    with open(DATA_DIR / "jobs.json", "r", encoding="utf-8") as f:
        jobs = json.load(f)
    with open(DATA_DIR / "skills.json", "r", encoding="utf-8") as f:
        skills = json.load(f)
    return RecommendationEngine(jobs), ChatBot(jobs, skills)