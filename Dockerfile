FROM python:3.12.7-slim

# cai dat libgl1-mesa-glx
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# thiet lap thu muc lam viec
WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD [ "python", "-m", "filter.app" ]

