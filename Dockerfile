# استخدام نسخة بايثون خفيفة
FROM python:3.11-slim

# إنشاء مستخدم غير جذري (شرط أمني في Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# تعيين مجلد العمل
WORKDIR /app

# نسخ ملف المتطلبات وتثبيتها
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع
COPY --chown=user . .

# Hugging Face يتطلب المنفذ 7860
EXPOSE 7860

# تشغيل الخادم
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]