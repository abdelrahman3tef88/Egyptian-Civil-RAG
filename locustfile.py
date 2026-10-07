# ==========================================================
# locustfile.py - اختبار حمل لـPOST /ask (streaming-aware)
# ==========================================================
#
# بنقرا الـstreaming response chunk-by-chunk باستخدام requests عادي
# (stream=True) وبنقيس بنفسنا، وبعدين بنسجّل 3 أحداث حقيقية في تقارير
# Locust لكل طلب - مفيش رقم مختلق:
#   1. "/ask"         -> زمن الاستجابة الكلي (Response Time)
#   2. "/ask_ttft"    -> الزمن لحد أول token (TTFT)
#   3. "/ask_tbt_avg" -> متوسط الزمن بين الـtokens (TBT)
#
# أرقام Response Time و P50 و P95 و P99 و Requests/sec بتطلع من إحصائيات
# Locust العادية على الحدث الأول، وTTFT/TBT من الحدثين التانيين.
#
# الـbenchmark الحقيقي (50 مستخدم):
#   locust -f locustfile.py --headless -u 50 -r 10 \
#     --run-time 1m --html reports/locust_report.html
#
# ملاحظة: مجلد reports/ لازم يكون موجود قبل توليد التقرير.

import statistics
import time

import requests
from locust import HttpUser, between, task

# سؤال حقيقي عن شروط صحة العقد في القانون المدني المصري.
REAL_QUESTION = "ما هي شروط صحة العقد في القانون المدني المصري؟"


def _fire(environment, name, response_time_ms, response_length=0, exception=None):
    """نسجّل حدث واحد بأرقام حقيقية اتقاست فعلًا (مش مختلقة)."""
    environment.events.request.fire(
        request_type="POST",
        name=name,
        response_time=response_time_ms,
        response_length=response_length,
        exception=exception,
        context={},
    )


class RAGUser(HttpUser):
    """مستخدم وهمي بيبعت أسئلة لـ /ask وبيقرا الـstream."""

    # وقت الانتظار بين كل طلب والتاني عشان نحاكي مستخدم حقيقي.
    wait_time = between(1, 3)

    @task
    def ask_streaming(self):
        # بنبعت POST /ask بالشكل الوحيد المتوقع من الخدمة.
        # مبنبعتش ground truth ولا metadata - السؤال بس.
        url = f"{self.host}/ask"

        request_start = time.perf_counter()
        first_token_time = None
        previous_token_time = None
        tbt_values = []
        chunks_received = 0
        text_bytes = 0

        try:
            # stream=True عشان نقرا الـchunks أول ما توصل من غير
            # ما نستنى الإجابة الكاملة.
            response = requests.post(
                url,
                json={"question": REAL_QUESTION},
                stream=True,
                timeout=300,
            )

            if not 200 <= response.status_code < 300:
                # الـservice رجّع status فاشل (مش إجابة) - نسجّل فشل.
                _fire(
                    self.environment,
                    "/ask",
                    (time.perf_counter() - request_start) * 1000.0,
                    exception=Exception(f"HTTP {response.status_code}"),
                )
                return

            # بنقرا الـstreaming body chunk-by-chunk وبنقيس التوقيت
            # بنفس المعادلات البسيطة بتاعة الـservice.
            for content in response.iter_content(chunk_size=None):
                if not content:
                    continue

                now = time.perf_counter()
                chunks_received += 1
                text_bytes += len(content)

                if first_token_time is None:
                    # أول chunk وصل = TTFT.
                    first_token_time = now
                else:
                    # الفرق عن الـchunk اللي قبله = TBT.
                    tbt_values.append(now - previous_token_time)

                previous_token_time = now

            response.close()

        except Exception as error:  # noqa: BLE001
            # أي عطل في الاتصال أو القراءة - نسجّله كفشل واضح.
            _fire(
                self.environment,
                "/ask",
                (time.perf_counter() - request_start) * 1000.0,
                exception=error,
            )
            return

        if first_token_time is None:
            # مفيش chunks أصلًا - ده فشل (مش صفر).
            _fire(
                self.environment,
                "/ask",
                (time.perf_counter() - request_start) * 1000.0,
                exception=Exception("no streamed chunks received"),
            )
            return

        total_ms = (time.perf_counter() - request_start) * 1000.0
        ttft_ms = (first_token_time - request_start) * 1000.0

        # 1) زمن الاستجابة الكلي -> Response Time و P50 و P95 و P99
        #    و Requests/sec كلها من إحصائيات Locust العادية.
        _fire(self.environment, "/ask", total_ms, response_length=text_bytes)

        # 2) TTFT كحدث منفصل عشان يطلع له percentiles في التقرير.
        _fire(self.environment, "/ask_ttft", ttft_ms)

        # 3) متوسط TBT للطلب ده كحدث منفصل.
        if tbt_values:
            _fire(
                self.environment,
                "/ask_tbt_avg",
                statistics.fmean(tbt_values) * 1000.0,
            )
