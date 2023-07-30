FROM python:3.10-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock","./"]

RUN pipenv install --deploy --system

ENV EXPERIMENT="Pistachio_Classifier"

COPY ["classify_service.py", "/model/Pistachio_Classifier_model.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "classify_service:app"]

