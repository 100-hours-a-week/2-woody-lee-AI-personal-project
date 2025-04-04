# Python 3.8-slim 기반 이미지 사용
FROM python:3.8-slim

WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY app/ app/

EXPOSE 8000

# 컨테이너 시작 시 FastAPI 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]