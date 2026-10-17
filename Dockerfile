FROM registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest
USER root
WORKDIR /app
# cryptography has no s390x wheel and builds from source against OpenSSL
RUN dnf install -y openssl-devel pkgconf-pkg-config gcc && dnf clean all
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY forest/ forest/
COPY scripts/ scripts/
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
USER ibm-user
CMD ["uvicorn", "forest.api.app:app", "--host", "127.0.0.1", "--port", "8000"]
